import contextlib
import dis
import functools
import logging
import os.path
import random
import re
import sys
import types
import unittest
from typing import List, Optional, Sequence, Union
from unittest.mock import patch
import torch
from torch import fx
from torch._dynamo.output_graph import OutputGraph
from . import config, eval_frame, optimize_assert, reset
from .bytecode_transformation import (
from .guards import CheckFunctionManager, GuardedCode
from .utils import same
def make_test_cls_with_patches(cls, cls_prefix, fn_suffix, *patches, xfail_prop=None):
    DummyTestClass = type(f'{cls_prefix}{cls.__name__}', cls.__bases__, {})
    DummyTestClass.__qualname__ = DummyTestClass.__name__
    for name in dir(cls):
        if name.startswith('test_'):
            fn = getattr(cls, name)
            if not callable(fn):
                setattr(DummyTestClass, name, getattr(cls, name))
                continue
            new_name = f'{name}{fn_suffix}'
            new_fn = _make_fn_with_patches(fn, *patches)
            new_fn.__name__ = new_name
            if xfail_prop is not None and hasattr(fn, xfail_prop):
                new_fn = unittest.expectedFailure(new_fn)
            setattr(DummyTestClass, new_name, new_fn)
        elif not hasattr(DummyTestClass, name):
            setattr(DummyTestClass, name, getattr(cls, name))
    return DummyTestClass