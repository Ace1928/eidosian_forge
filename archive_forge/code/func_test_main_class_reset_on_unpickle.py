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
@unittest.skipIf(__name__ == '__main__', 'Test cannot run as when module is __main__')
def test_main_class_reset_on_unpickle(self):
    mp = get_context('spawn')
    proc = mp.Process(target=check_main_class_reset_on_unpickle)
    proc.start()
    proc.join(timeout=60)
    self.assertEqual(proc.exitcode, 0)