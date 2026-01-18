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
def test_dynamic_class_reset_on_unpickle_new_proc(self):

    class Klass:
        classvar = None
    saved = dumps(Klass)
    mp = get_context('spawn')
    proc = mp.Process(target=check_unpickle_dyn_class_new_proc, args=(saved,))
    proc.start()
    proc.join(timeout=60)
    self.assertEqual(proc.exitcode, 0)