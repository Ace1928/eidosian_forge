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
def check_unpickle_dyn_class_new_proc(saved):
    Klass = loads(saved)
    assert Klass.classvar != 100
    Klass.classvar = 100
    _check_dyn_class(Klass, saved)