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
def set_evaluate(self, strategy: Union[str, IntervalStrategy]='no', steps: int=500, batch_size: int=8, accumulation_steps: Optional[int]=None, delay: Optional[float]=None, loss_only: bool=False, jit_mode: bool=False):
    """
        A method that regroups all arguments linked to evaluation.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
                The evaluation strategy to adopt during training. Possible values are:

                    - `"no"`: No evaluation is done during training.
                    - `"steps"`: Evaluation is done (and logged) every `steps`.
                    - `"epoch"`: Evaluation is done at the end of each epoch.

                Setting a `strategy` different from `"no"` will set `self.do_eval` to `True`.
            steps (`int`, *optional*, defaults to 500):
                Number of update steps between two evaluations if `strategy="steps"`.
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for evaluation.
            accumulation_steps (`int`, *optional*):
                Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU.
                If left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster
                but requires more memory).
            delay (`float`, *optional*):
                Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
                evaluation_strategy.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.
            jit_mode (`bool`, *optional*):
                Whether or not to use PyTorch jit trace for inference.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_evaluate(strategy="steps", steps=100)
        >>> args.eval_steps
        100
        ```
        """
    self.evaluation_strategy = IntervalStrategy(strategy)
    if self.evaluation_strategy == IntervalStrategy.STEPS and steps == 0:
        raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
    self.do_eval = self.evaluation_strategy != IntervalStrategy.NO
    self.eval_steps = steps
    self.per_device_eval_batch_size = batch_size
    self.eval_accumulation_steps = accumulation_steps
    self.eval_delay = delay
    self.prediction_loss_only = loss_only
    self.jit_mode_eval = jit_mode
    return self