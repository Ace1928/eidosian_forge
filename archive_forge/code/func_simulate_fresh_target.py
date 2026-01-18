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
@contextlib.contextmanager
def simulate_fresh_target(self):
    hwstr = 'cpu'
    dispatcher_cls = resolve_dispatcher_from_str(hwstr)
    old_descr = dispatcher_cls.targetdescr
    dispatcher_cls.targetdescr = type(dispatcher_cls.targetdescr)(hwstr)
    try:
        yield
    finally:
        dispatcher_cls.targetdescr = old_descr