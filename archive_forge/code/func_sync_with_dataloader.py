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
def sync_with_dataloader(self) -> bool:
    """Returns whether the gradients should be synced at the end of the dataloader iteration and the number of total steps reset"""
    return self.plugin_kwargs.get('sync_with_dataloader', True)