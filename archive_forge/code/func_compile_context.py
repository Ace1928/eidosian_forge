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
@contextmanager
def compile_context(context: CompileContext):
    old_context = getattr(_TLS, 'compile_context', None)
    _TLS.compile_context = context
    try:
        yield context
    finally:
        _TLS.compile_context = old_context