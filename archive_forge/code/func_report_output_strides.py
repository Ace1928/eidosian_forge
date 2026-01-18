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
@contextlib.contextmanager
def report_output_strides():
    tc = TracingContext.try_get()
    if tc is None:
        yield None
        return
    old_output_strides = tc.output_strides
    tc.output_strides = []
    try:
        yield tc.output_strides
    finally:
        tc.output_strides = old_output_strides