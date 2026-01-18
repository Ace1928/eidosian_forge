import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
def set_dataloader(self, train_batch_size: int=8, eval_batch_size: int=8, drop_last: bool=False, num_workers: int=0, pin_memory: bool=True, persistent_workers: bool=False, prefetch_factor: Optional[int]=None, auto_find_batch_size: bool=False, ignore_data_skip: bool=False, sampler_seed: Optional[int]=None):
    """
        A method that regroups all arguments linked to the dataloaders creation.

        Args:
            drop_last (`bool`, *optional*, defaults to `False`):
                Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch
                size) or not.
            num_workers (`int`, *optional*, defaults to 0):
                Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in
                the main process.
            pin_memory (`bool`, *optional*, defaults to `True`):
                Whether you want to pin memory in data loaders or not. Will default to `True`.
            persistent_workers (`bool`, *optional*, defaults to `False`):
                If True, the data loader will not shut down the worker processes after a dataset has been consumed
                once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training,
                but will increase RAM usage. Will default to `False`.
            prefetch_factor (`int`, *optional*):
                Number of batches loaded in advance by each worker.
                2 means there will be a total of 2 * num_workers batches prefetched across all workers.
            auto_find_batch_size (`bool`, *optional*, defaults to `False`)
                Whether to find a batch size that will fit into memory automatically through exponential decay,
                avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)
            ignore_data_skip (`bool`, *optional*, defaults to `False`):
                When resuming training, whether or not to skip the epochs and batches to get the data loading at the
                same stage as in the previous training. If set to `True`, the training will begin faster (as that
                skipping step can take a long time) but will not yield the same results as the interrupted training
                would have.
            sampler_seed (`int`, *optional*):
                Random seed to be used with data samplers. If not set, random generators for data sampling will use the
                same seed as `self.seed`. This can be used to ensure reproducibility of data sampling, independent of
                the model seed.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_dataloader(train_batch_size=16, eval_batch_size=64)
        >>> args.per_device_train_batch_size
        16
        ```
        """
    self.per_device_train_batch_size = train_batch_size
    self.per_device_eval_batch_size = eval_batch_size
    self.dataloader_drop_last = drop_last
    self.dataloader_num_workers = num_workers
    self.dataloader_pin_memory = pin_memory
    self.dataloader_persistent_workers = persistent_workers
    self.dataloader_prefetch_factor = prefetch_factor
    self.auto_find_batch_size = auto_find_batch_size
    self.ignore_data_skip = ignore_data_skip
    self.data_seed = sampler_seed
    return self