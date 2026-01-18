from __future__ import annotations
import contextlib
import functools
import json
import math
import os
import re
import shutil
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from types import MethodType
from typing import Any, Callable, Union
import torch
import torch.utils.hooks as hooks
from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
from .data_loader import DataLoaderDispatcher, prepare_data_loader, skip_first_batches
from .hooks import AlignDevicesHook
from .logging import get_logger
from .optimizer import AcceleratedOptimizer
from .scheduler import AcceleratedScheduler
from .state import AcceleratorState, GradientState, PartialState
from .tracking import LOGGER_TYPE_TO_CLASS, GeneralTracker, filter_trackers
from .utils import (
from .utils.constants import FSDP_PYTORCH_VERSION
from .utils.modeling import get_state_dict_offloaded_model
from .utils.other import is_compiled_module
from torch.distributed.algorithms.join import Join
@contextmanager
def join_uneven_inputs(self, joinables, even_batches=None):
    """
        A context manager that facilitates distributed training or evaluation on uneven inputs, which acts as a wrapper
        around `torch.distributed.algorithms.join`. This is useful when the total batch size does not evenly divide the
        length of the dataset.

        Args:
            joinables (`list[torch.distributed.algorithms.Joinable]`):
                A list of models or optimizers that subclass `torch.distributed.algorithms.Joinable`. Most commonly, a
                PyTorch Module that was prepared with `Accelerator.prepare` for DistributedDataParallel training.
            even_batches (`bool`, *optional*)
                If set, this will override the value of `even_batches` set in the `Accelerator`. If it is not provided,
                the default `Accelerator` value wil be used.

        <Tip warning={true}>

        `join_uneven_inputs` is only supported for Distributed Data Parallel training on multiple GPUs. For any other
        configuration, this method will have no effect.

        </Tip>

        <Tip warning={true}>

        Overidding `even_batches` will not affect iterable-style data loaders.

        </Tip>

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(even_batches=True)
        >>> ddp_model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

        >>> with accelerator.join_uneven_inputs([ddp_model], even_batches=False):
        ...     for input, output in dataloader:
        ...         outputs = model(input)
        ...         loss = loss_func(outputs)
        ...         loss.backward()
        ...         optimizer.step()
        ...         optimizer.zero_grad()
        ```
        """
    if self.distributed_type in (DistributedType.MULTI_GPU, DistributedType.MULTI_NPU, DistributedType.MULTI_XPU):
        dl_even_batches_values = []
        if even_batches is not None:
            iterable_dl_seen = False
            for dl_idx, dl in enumerate(self._dataloaders):
                if isinstance(dl, DataLoaderDispatcher):
                    iterable_dl_seen = True
                    continue
                dl_even_batches_values.append((dl_idx, dl.batch_sampler.even_batches))
                dl.batch_sampler.even_batches = even_batches
            if iterable_dl_seen:
                warnings.warn('Overridding even_batches is only supported for map-style datasets, yet some dataloaders given were iterable')
        else:
            even_batches = self.even_batches
        enable_join = False if even_batches else True
        try:
            with Join(joinables, enable=enable_join, throw_on_early_termination=False):
                yield
        finally:
            for dl_idx, even_batches_value in dl_even_batches_values:
                self._dataloaders[dl_idx].batch_sampler.even_batches = even_batches_value
    else:
        if self.distributed_type != DistributedType.NO:
            warnings.warn('Joining uneven inputs is only supported for multi-GPU training, as a result `join_uneven_inputs` will have no effect.')
        with contextlib.nullcontext(joinables):
            yield