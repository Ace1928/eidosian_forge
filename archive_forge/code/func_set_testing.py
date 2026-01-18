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
def set_testing(self, batch_size: int=8, loss_only: bool=False, jit_mode: bool=False):
    """
        A method that regroups all basic arguments linked to testing on a held-out dataset.

        <Tip>

        Calling this method will automatically set `self.do_predict` to `True`.

        </Tip>

        Args:
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for testing.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.
            jit_mode (`bool`, *optional*):
                Whether or not to use PyTorch jit trace for inference.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_testing(batch_size=32)
        >>> args.per_device_eval_batch_size
        32
        ```
        """
    self.do_predict = True
    self.per_device_eval_batch_size = batch_size
    self.prediction_loss_only = loss_only
    self.jit_mode_eval = jit_mode
    return self