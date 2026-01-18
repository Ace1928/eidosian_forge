from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
def make_obj(base, array_priority=False, array_ufunc=False, alleged_module='__main__'):
    class_namespace = {'__array__': array_impl}
    if array_priority is not False:
        class_namespace['__array_priority__'] = array_priority
    for op in ops:
        class_namespace['__{0}__'.format(op)] = op_impl
        class_namespace['__r{0}__'.format(op)] = rop_impl
        class_namespace['__i{0}__'.format(op)] = iop_impl
    if array_ufunc is not False:
        class_namespace['__array_ufunc__'] = array_ufunc
    eval_namespace = {'base': base, 'class_namespace': class_namespace, '__name__': alleged_module}
    MyType = eval("type('MyType', (base,), class_namespace)", eval_namespace)
    if issubclass(MyType, np.ndarray):
        return np.arange(3, 7).reshape(2, 2).view(MyType)
    else:
        return MyType()