from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
@property
def mixed_precision(self):
    if self.distributed_type == DistributedType.DEEPSPEED:
        config = self.deepspeed_plugin.deepspeed_config
        if config.get('fp16', {}).get('enabled', False):
            mixed_precision = 'fp16'
        elif config.get('bf16', {}).get('enabled', False):
            mixed_precision = 'bf16'
        else:
            mixed_precision = 'no'
    else:
        mixed_precision = self._mixed_precision
    return mixed_precision