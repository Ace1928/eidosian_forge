import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def interpreter_exec(self, interpreter=None, command=None):
    """gdb command ~= 'interpreter-exec'"""
    if interpreter is None:
        raise ValueError('interpreter cannot be None')
    if command is None:
        raise ValueError('command cannot be None')
    cmd = f'-interpreter-exec {interpreter} "{command}"'
    self._run_command(cmd, expect='\\^(done|error).*\\r\\n')