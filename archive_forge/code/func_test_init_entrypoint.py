import sys
from unittest import mock
import types
import warnings
import unittest
import os
import subprocess
import threading
from numba import config, njit
from numba.tests.support import TestCase
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
def test_init_entrypoint(self):
    mod = mock.Mock(__name__='_test_numba_extension')
    try:
        sys.modules[mod.__name__] = mod
        my_entrypoint = importlib_metadata.EntryPoint('init', '_test_numba_extension:init_func', 'numba_extensions')
        with mock.patch.object(importlib_metadata, 'entry_points', return_value={'numba_extensions': (my_entrypoint,)}):
            from numba.core import entrypoints
            entrypoints._already_initialized = False
            entrypoints.init_all()
            mod.init_func.assert_called_once()
            entrypoints.init_all()
            mod.init_func.assert_called_once()
    finally:
        if mod.__name__ in sys.modules:
            del sys.modules[mod.__name__]