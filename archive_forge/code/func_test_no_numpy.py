import os
import subprocess
import sys
import threading
import json
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest import mock
import unittest
from numba.tests.support import TestCase, linux_only
import numba.misc.numba_sysinfo as nsi
from numba.tests.gdb_support import needs_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
from numba.misc.numba_gdbinfo import _GDBTestWrapper
def test_no_numpy(self):

    def mock_fn(self):
        return CompletedProcess('NO NUMPY', 1)
    with mock.patch.object(_GDBTestWrapper, 'check_numpy', mock_fn):
        collected = collect_gdbinfo()
        self.assertEqual(collected.np_ver, 'No NumPy support')
        self.assertEqual(collected.py_ver, '3.2')
        self.assertIn('Partial', collected.supported)