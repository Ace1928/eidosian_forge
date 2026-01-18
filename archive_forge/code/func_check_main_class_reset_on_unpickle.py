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
def check_main_class_reset_on_unpickle():
    glbs = runpy.run_module('numba.tests.cloudpickle_main_class', run_name='__main__')
    Klass = glbs['Klass']
    assert Klass.__module__ == '__main__'
    assert Klass.classvar != 100
    saved = dumps(Klass)
    Klass.classvar = 100
    _check_dyn_class(Klass, saved)