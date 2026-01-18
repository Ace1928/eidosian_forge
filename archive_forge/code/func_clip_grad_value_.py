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
def clip_grad_value_(self, parameters, clip_value):
    """
        Should be used in place of `torch.nn.utils.clip_grad_value_`.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=2)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     output = model(input)
        ...     loss = loss_func(output, target)
        ...     accelerator.backward(loss)
        ...     if accelerator.sync_gradients:
        ...         accelerator.clip_grad_value_(model.parameters(), clip_value)
        ...     optimizer.step()
        ```
        """
    if self.distributed_type in [DistributedType.DEEPSPEED, DistributedType.FSDP]:
        raise Exception('DeepSpeed and FSDP  do not support `clip_grad_value_`. Use `clip_grad_norm_` instead.')
    self.unscale_gradients()
    torch.nn.utils.clip_grad_value_(parameters, clip_value)