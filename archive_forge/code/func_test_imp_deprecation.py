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
def test_imp_deprecation(self):
    """
        The imp module was deprecated in v3.4 in favour of importlib
        """
    code = 'if 1:\n            import pickle\n            import warnings\n            with warnings.catch_warnings(record=True) as w:\n                warnings.simplefilter(\'always\', DeprecationWarning)\n                from numba import njit\n                @njit\n                def foo(x):\n                    return x + 1\n                foo(1)\n                serialized_foo = pickle.dumps(foo)\n            for x in w:\n                if \'serialize.py\' in x.filename:\n                    assert "the imp module is deprecated" not in x.msg\n        '
    subprocess.check_call([sys.executable, '-c', code])