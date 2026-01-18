import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def stack_list_variables(self, print_values=1):
    """gdb command ~= 'info locals'"""
    assert isinstance(print_values, int) and print_values in (0, 1, 2)
    cmd = f'-stack-list-variables {print_values}'
    self._run_command(cmd, expect='\\^done,.*\\r\\n')