from __future__ import annotations
import contextlib
import dataclasses
import enum
import functools
import logging
import threading
import traceback
import unittest.mock
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
import torch
from torch.utils import _pytree as pytree
from torch.utils._traceback import CapturedTraceback
from torch.utils.weak import WeakTensorKeyDictionary
@staticmethod
def set_current_loc(filename, lineno, frame_name):
    TracingContext.get().loc_in_frame = traceback.FrameSummary(filename, lineno, frame_name)