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
@on_main_process
def init_trackers(self, project_name: str, config: dict | None=None, init_kwargs: dict | None={}):
    """
        Initializes a run for all trackers stored in `self.log_with`, potentially with starting configurations

        Args:
            project_name (`str`):
                The name of the project. All trackers will save their data based on this
            config (`dict`, *optional*):
                Optional starting configuration to be logged.
            init_kwargs (`dict`, *optional*):
                A nested dictionary of kwargs to be passed to a specific tracker's `__init__` function. Should be
                formatted like so:
                ```python
                {"wandb": {"tags": ["tag_a", "tag_b"]}}
                ```

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(log_with="tensorboard")
        >>> accelerator.init_trackers(
        ...     project_name="my_project",
        ...     config={"learning_rate": 0.001, "batch_size": 32},
        ...     init_kwargs={"tensorboard": {"flush_secs": 60}},
        ... )
        ```
        """
    for tracker in self.log_with:
        if issubclass(type(tracker), GeneralTracker):
            self.trackers.append(tracker)
        else:
            tracker_init = LOGGER_TYPE_TO_CLASS[str(tracker)]
            if tracker_init.requires_logging_directory:
                self.trackers.append(tracker_init(project_name, self.logging_dir, **init_kwargs.get(str(tracker), {})))
            else:
                self.trackers.append(tracker_init(project_name, **init_kwargs.get(str(tracker), {})))
    if config is not None:
        for tracker in self.trackers:
            tracker.store_init_configuration(config)