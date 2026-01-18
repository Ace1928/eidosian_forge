import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def stack_list_arguments(self, print_values=1, low_frame=0, high_frame=0):
    """gdb command ~= 'info args'"""
    for x in (print_values, low_frame, high_frame):
        assert isinstance(x, int) and x in (0, 1, 2)
    cmd = f'-stack-list-arguments {print_values} {low_frame} {high_frame}'
    self._run_command(cmd, expect='\\^done,.*\\r\\n')