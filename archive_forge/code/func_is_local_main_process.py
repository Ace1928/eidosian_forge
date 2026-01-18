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
def is_local_main_process(self) -> bool:
    """Returns whether the current process is the main process on the local node"""
    return PartialState().is_local_main_process