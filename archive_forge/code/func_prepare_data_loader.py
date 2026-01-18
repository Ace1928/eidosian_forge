import collections
import logging
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from packaging.version import Version
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import (
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train._internal import session
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.session import get_accelerator, set_accelerator
from ray.util.annotations import Deprecated, PublicAPI
def prepare_data_loader(self, data_loader: torch.utils.data.DataLoader, add_dist_sampler: bool=True, move_to_device: bool=True, auto_transfer: bool=False) -> torch.utils.data.DataLoader:
    """Prepares DataLoader for distributed execution.

        This allows you to use the same exact code regardless of number of
        workers or the device type being used (CPU, GPU).

        Args:
            data_loader (torch.utils.data.DataLoader): The DataLoader to
                prepare.
            add_dist_sampler: Whether to add a DistributedSampler to
                the provided DataLoader.
            move_to_device: If set, automatically move the data
                returned by the data loader to the correct device.
            auto_transfer: (Experimental) If set and device is GPU, another CUDA stream
                is created to automatically copy data from host (CPU) memory
                to device (GPU) memory (the default CUDA stream still runs the
                training procedure). If device is CPU, it will be disabled
                regardless of the setting. This configuration will be ignored
                if ``move_to_device`` is False.
        """
    world_size = session.get_world_size()
    world_rank = session.get_world_rank()
    if world_size > 1 and (not isinstance(data_loader.sampler, DistributedSampler)) and (not (hasattr(data_loader, 'dataset') and isinstance(data_loader.dataset, IterableDataset))) and add_dist_sampler:

        def with_sampler(loader):
            shuffle = not isinstance(loader.sampler, SequentialSampler)

            def seeded_worker_init_fn(worker_init_fn: Optional[Callable[[int], None]]):

                def wrapper(worker_id: int):
                    worker_seed = torch.initial_seed() % 2 ** 32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)
                    if worker_init_fn:
                        worker_init_fn(worker_id)
                return wrapper
            worker_init_fn: Optional[Callable[[int], None]] = loader.worker_init_fn
            generator: Optional[torch.Generator] = loader.generator
            if self._seed is not None:
                worker_init_fn = seeded_worker_init_fn(worker_init_fn)
                generator = torch.Generator()
                generator.manual_seed(self._seed)
            using_default_sampler = isinstance(loader.sampler, (SequentialSampler, RandomSampler))
            if not using_default_sampler and world_rank == 0:
                logger.warn(f'The {loader.sampler.__class__.__name__} will be overwritten with a DistributedSampler. You can disable this by setting `with_sampler` to False in `prepare_data_loader`.')
            data_loader_args = {'dataset': loader.dataset, 'batch_size': loader.batch_size, 'shuffle': False, 'num_workers': loader.num_workers, 'collate_fn': loader.collate_fn, 'pin_memory': loader.pin_memory, 'drop_last': loader.drop_last, 'timeout': loader.timeout, 'worker_init_fn': worker_init_fn, 'generator': generator, 'sampler': DistributedSampler(loader.dataset, shuffle=shuffle)}
            return DataLoader(**data_loader_args)
        data_loader = with_sampler(data_loader)
    if move_to_device:
        device = get_device()
        data_loader = _WrappedDataLoader(data_loader, device, auto_transfer)
    return data_loader