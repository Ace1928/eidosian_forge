import inspect
from typing import IO, Any, Dict, Iterable, Optional, Union, cast
from lightning_utilities import apply_to_collection
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing_extensions import Self
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from pytorch_lightning.core.hooks import DataHooks
from pytorch_lightning.core.mixins import HyperparametersMixin
from pytorch_lightning.core.saving import _load_from_checkpoint
from pytorch_lightning.utilities.model_helpers import _restricted_classmethod
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
@classmethod
def from_datasets(cls, train_dataset: Optional[Union[Dataset, Iterable[Dataset]]]=None, val_dataset: Optional[Union[Dataset, Iterable[Dataset]]]=None, test_dataset: Optional[Union[Dataset, Iterable[Dataset]]]=None, predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]]=None, batch_size: int=1, num_workers: int=0, **datamodule_kwargs: Any) -> 'LightningDataModule':
    """Create an instance from torch.utils.data.Dataset.

        Args:
            train_dataset: Optional dataset or iterable of datasets to be used for train_dataloader()
            val_dataset: Optional dataset or iterable of datasets to be used for val_dataloader()
            test_dataset: Optional dataset or iterable of datasets to be used for test_dataloader()
            predict_dataset: Optional dataset or iterable of datasets to be used for predict_dataloader()
            batch_size: Batch size to use for each dataloader. Default is 1. This parameter gets forwarded to the
                ``__init__`` if the datamodule has such a name defined in its signature.
            num_workers: Number of subprocesses to use for data loading. 0 means that the
                data will be loaded in the main process. Number of CPUs available. This parameter gets forwarded to the
                ``__init__`` if the datamodule has such a name defined in its signature.
            **datamodule_kwargs: Additional parameters that get passed down to the datamodule's ``__init__``.

        """

    def dataloader(ds: Dataset, shuffle: bool=False) -> DataLoader:
        shuffle &= not isinstance(ds, IterableDataset)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def train_dataloader() -> TRAIN_DATALOADERS:
        return apply_to_collection(train_dataset, Dataset, dataloader, shuffle=True)

    def val_dataloader() -> EVAL_DATALOADERS:
        return apply_to_collection(val_dataset, Dataset, dataloader)

    def test_dataloader() -> EVAL_DATALOADERS:
        return apply_to_collection(test_dataset, Dataset, dataloader)

    def predict_dataloader() -> EVAL_DATALOADERS:
        return apply_to_collection(predict_dataset, Dataset, dataloader)
    candidate_kwargs = {'batch_size': batch_size, 'num_workers': num_workers}
    accepted_params = inspect.signature(cls.__init__).parameters
    accepts_kwargs = any((param.kind == param.VAR_KEYWORD for param in accepted_params.values()))
    if accepts_kwargs:
        special_kwargs = candidate_kwargs
    else:
        accepted_param_names = set(accepted_params)
        accepted_param_names.discard('self')
        special_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted_param_names}
    datamodule = cls(**datamodule_kwargs, **special_kwargs)
    if train_dataset is not None:
        datamodule.train_dataloader = train_dataloader
    if val_dataset is not None:
        datamodule.val_dataloader = val_dataloader
    if test_dataset is not None:
        datamodule.test_dataloader = test_dataloader
    if predict_dataset is not None:
        datamodule.predict_dataloader = predict_dataloader
    return datamodule