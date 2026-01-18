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
def set_lr_scheduler(self, name: Union[str, SchedulerType]='linear', num_epochs: float=3.0, max_steps: int=-1, warmup_ratio: float=0, warmup_steps: int=0):
    """
        A method that regroups all arguments linked to the learning rate scheduler and its hyperparameters.

        Args:
            name (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
                The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
            num_epochs(`float`, *optional*, defaults to 3.0):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training).
            max_steps (`int`, *optional*, defaults to -1):
                If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
                For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
                `max_steps` is reached.
            warmup_ratio (`float`, *optional*, defaults to 0.0):
                Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
            warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of
                `warmup_ratio`.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_lr_scheduler(name="cosine", warmup_ratio=0.05)
        >>> args.warmup_ratio
        0.05
        ```
        """
    self.lr_scheduler_type = SchedulerType(name)
    self.num_train_epochs = num_epochs
    self.max_steps = max_steps
    self.warmup_ratio = warmup_ratio
    self.warmup_steps = warmup_steps
    return self