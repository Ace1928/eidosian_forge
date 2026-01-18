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
def test_invalid_binary(self):

    def mock_fn(self):
        return CompletedProcess('INVALID_BINARY', 1)
    with mock.patch.object(_GDBTestWrapper, 'check_launch', mock_fn):
        info = collect_gdbinfo()
        self.assertIn('Testing gdb binary failed.', info.binary_loc)
        self.assertIn("gdb at 'PATH_TO_GDB' does not appear to work", info.binary_loc)