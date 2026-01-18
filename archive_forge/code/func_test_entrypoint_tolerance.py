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
def test_entrypoint_tolerance(self):
    mod = mock.Mock(__name__='_test_numba_bad_extension')
    mod.configure_mock(**{'init_func.side_effect': ValueError('broken')})
    try:
        sys.modules[mod.__name__] = mod
        my_entrypoint = importlib_metadata.EntryPoint('init', '_test_numba_bad_extension:init_func', 'numba_extensions')
        with mock.patch.object(importlib_metadata, 'entry_points', return_value={'numba_extensions': (my_entrypoint,)}):
            from numba.core import entrypoints
            entrypoints._already_initialized = False
            with warnings.catch_warnings(record=True) as w:
                entrypoints.init_all()
            bad_str = "Numba extension module '_test_numba_bad_extension'"
            for x in w:
                if bad_str in str(x):
                    break
            else:
                raise ValueError('Expected warning message not found')
            mod.init_func.assert_called_once()
    finally:
        if mod.__name__ in sys.modules:
            del sys.modules[mod.__name__]