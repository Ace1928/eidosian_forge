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
def test_call_closure_with_globals_nopython(self):
    self.check_call_closure_with_globals(nopython=True)