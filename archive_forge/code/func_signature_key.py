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
def signature_key(self):
    annotation = self.param.annotation
    if 'Tensor' in annotation:
        return self.value.dtype
    elif annotation == 'bool':
        return 'i1'
    elif annotation == 'float':
        return 'fp32'
    else:
        return JITFunction._key_of(self.value)