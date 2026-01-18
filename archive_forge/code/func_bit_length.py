import atexit
import builtins
import functools
import inspect
import os
import operator
import timeit
import math
import sys
import traceback
import weakref
import warnings
import threading
import contextlib
import typing as _tp
from types import ModuleType
from importlib import import_module
import numpy as np
from inspect import signature as pysignature # noqa: F401
from inspect import Signature as pySignature # noqa: F401
from inspect import Parameter as pyParameter # noqa: F401
from numba.core.config import (PYVERSION, MACHINE_BITS, # noqa: F401
from numba.core import config
from numba.core import types
from collections.abc import Mapping, Sequence, MutableSet, MutableMapping
def bit_length(intval):
    """
    Return the number of bits necessary to represent integer `intval`.
    """
    assert isinstance(intval, int)
    if intval >= 0:
        return len(bin(intval)) - 2
    else:
        return len(bin(-intval - 1)) - 2