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
def specialization_key(self):
    assert not self.param.do_not_specialize
    try:
        return (self.value.data_ptr() % JITFunction.divisibility == 0,)
    except AttributeError:
        pass
    if isinstance(self.value, int):
        return (self.value % JITFunction.divisibility == 0, self.value % JITFunction.divisibility_8 == 0, self.value == 1)
    return (False,)