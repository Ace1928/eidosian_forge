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
def test_dynamic_class_reset_on_unpickle(self):

    class Klass:
        classvar = None

    def mutator():
        Klass.classvar = 100

    def check():
        self.assertEqual(Klass.classvar, 100)
    saved = dumps(Klass)
    mutator()
    check()
    loads(saved)
    check()
    loads(saved)
    check()