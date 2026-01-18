import contextlib
import gc
import pickle
import runpy
import subprocess
import sys
import unittest
from multiprocessing import get_context
import numba
from numba.core.errors import TypingError
from numba.tests.support import TestCase
from numba.core.target_extension import resolve_dispatcher_from_str
from numba.cloudpickle import dumps, loads
def issue_7356():
    with numba.objmode(before='intp'):
        DynClass.a = 100
        before = DynClass.a
    with numba.objmode(after='intp'):
        after = DynClass.a
    return (before, after)