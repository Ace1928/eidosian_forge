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
def end_of_dataloader(self) -> bool:
    """Returns whether we have reached the end of the current dataloader"""
    if not self.in_dataloader:
        return False
    return self.active_dataloader.end_of_dataloader