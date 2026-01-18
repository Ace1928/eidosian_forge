from __future__ import annotations, division
import ast
import functools
import hashlib
import inspect
import os
import textwrap
from collections import defaultdict, namedtuple
from functools import cached_property
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast, overload
from .._C.libtriton.triton import TMAInfos
from ..common.backend import get_backend, get_cuda_version_key
from .interpreter import InterpretedFunction
def is_divisible_by_16(x):
    if hasattr(x, 'data_ptr'):
        return x.data_ptr() % JITFunction.divisibility == 0
    elif isinstance(x, int):
        return x % JITFunction.divisibility == 0
    if x is None:
        return True
    return False