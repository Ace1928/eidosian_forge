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
def check_call_closure_with_globals(self, **jit_args):
    from .serialize_usecases import closure_with_globals
    inner = closure_with_globals(3.0, **jit_args)
    self.run_with_protocols(self.check_call, inner, 7.0, (4.0,))