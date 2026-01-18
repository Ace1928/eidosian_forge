import logging
import os
import shutil
from tempfile import TemporaryDirectory
from typing import Iterator, Optional, Type
from torch.utils.data import DataLoader, Dataset, IterableDataset
import ray
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.data.iterator import _IterableFromIterator
from ray.train import Checkpoint
from ray.util import PublicAPI
@PublicAPI(stability='beta')
def prepare_trainer(trainer: 'Trainer') -> 'Trainer':
    """Prepare your HuggingFace Transformer Trainer for Ray Train.

    This utility function enable the trainer integrates with Ray Data Integration.
    Internally, it overrides the `get_train_dataloader` and `get_eval_dataloader`
    methods and inject the data integration logics if the `train_dataset` and
    `eval_dataset` are Ray Data Iterables.
    """
    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise TRANSFORMERS_IMPORT_ERROR
    base_trainer_class: Type[transformers.trainer.Trainer] = trainer.__class__

    class RayTransformersTrainer(base_trainer_class):
        """A Wrapper of `transformers.Trainer` for Ray Data Integration."""

        def get_train_dataloader(self) -> DataLoader:
            if isinstance(self.train_dataset, _IterableFromIterator):
                dataset = RayTorchIterableDataset(self.train_dataset)
                return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
            else:
                return super().get_train_dataloader()

        def get_eval_dataloader(self, eval_dataset: Optional[Dataset]=None) -> DataLoader:
            if eval_dataset is None:
                eval_dataset = self.eval_dataset
            if isinstance(eval_dataset, _IterableFromIterator):
                dataset = RayTorchIterableDataset(eval_dataset)
                return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
            else:
                return super().get_eval_dataloader(eval_dataset)
    trainer.__class__ = RayTransformersTrainer
    record_extra_usage_tag(TagKey.TRAIN_TRANSFORMERS_PREPARE_TRAINER, '1')
    return trainer