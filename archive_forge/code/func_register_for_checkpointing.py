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
def register_for_checkpointing(self, *objects):
    """
        Makes note of `objects` and will save or load them in during `save_state` or `load_state`.

        These should be utilized when the state is being loaded or saved in the same script. It is not designed to be
        used in different scripts.

        <Tip>

        Every `object` must have a `load_state_dict` and `state_dict` function to be stored.

        </Tip>

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume `CustomObject` has a `state_dict` and `load_state_dict` function.
        >>> obj = CustomObject()
        >>> accelerator.register_for_checkpointing(obj)
        >>> accelerator.save_state("checkpoint.pt")
        ```
        """
    invalid_objects = []
    for obj in objects:
        if not hasattr(obj, 'state_dict') or not hasattr(obj, 'load_state_dict'):
            invalid_objects.append(obj)
    if len(invalid_objects) > 0:
        err = 'All `objects` must include a `state_dict` and `load_state_dict` function to be stored. The following inputs are invalid:'
        for index, obj in enumerate(invalid_objects):
            err += f'\n\t- Item at index {index}, `{get_pretty_name(obj)}`'
        raise ValueError(err)
    self._custom_objects.extend(objects)