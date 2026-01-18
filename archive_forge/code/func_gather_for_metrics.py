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
def gather_for_metrics(self, input_data):
    """
        Gathers `input_data` and potentially drops duplicates in the last batch if on a distributed system. Should be
        used for gathering the inputs and targets for metric calculation.

        Args:
            input (`torch.Tensor`, `object`, a nested tuple/list/dictionary of `torch.Tensor`, or a nested tuple/list/dictionary of `object`):
                The tensors or objects for calculating metrics across all processes

        Example:

        ```python
        >>> # Assuming two processes, with a batch size of 5 on a dataset with 9 samples
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> dataloader = torch.utils.data.DataLoader(range(9), batch_size=5)
        >>> dataloader = accelerator.prepare(dataloader)
        >>> batch = next(iter(dataloader))
        >>> gathered_items = accelerator.gather_for_metrics(batch)
        >>> len(gathered_items)
        9
        ```
        """
    try:
        recursively_apply(lambda x: x, input_data, error_on_other_type=True)
        all_tensors = True
    except TypeError:
        all_tensors = False
    if not all_tensors:
        data = gather_object(input_data)
    else:
        data = self.gather(input_data)
    try:
        if self.gradient_state.end_of_dataloader:
            if self.gradient_state.remainder == -1:
                logger.info('The used dataset had no length, returning gathered tensors. You should drop the remainder yourself.')
                return data
            elif self.gradient_state.remainder > 0:

                def _adjust_samples(tensor):
                    return tensor[:self.gradient_state.remainder]
                return recursively_apply(_adjust_samples, data)
            else:
                return data
        else:
            return data
    except Exception:
        return data