from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def addr2line(self, addrq):
    """
        Get the line number for a given bytecode offset

        Analogous to PyCode_Addr2Line; translated from pseudocode in
        Objects/lnotab_notes.txt
        """
    co_lnotab = self.pyop_field('co_lnotab').proxyval(set())
    lineno = int_from_int(self.field('co_firstlineno'))
    addr = 0
    for addr_incr, line_incr in zip(co_lnotab[::2], co_lnotab[1::2]):
        addr += ord(addr_incr)
        if addr > addrq:
            return lineno
        lineno += ord(line_incr)
    return lineno