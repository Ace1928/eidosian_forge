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
def test_call_without_sig(self):
    from .serialize_usecases import add_without_sig
    self.run_with_protocols(self.check_call, add_without_sig, 5, (1, 4))
    self.run_with_protocols(self.check_call, add_without_sig, 5.5, (1.2, 4.3))
    self.run_with_protocols(self.check_call, add_without_sig, 'abc', ('a', 'bc'))