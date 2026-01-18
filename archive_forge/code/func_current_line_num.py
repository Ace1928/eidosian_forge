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
def current_line_num(self):
    """Get current line number as an integer (1-based)

        Translated from PyFrame_GetLineNumber and PyCode_Addr2Line

        See Objects/lnotab_notes.txt
        """
    if self.is_optimized_out():
        return None
    f_trace = self.field('f_trace')
    if long(f_trace) != 0:
        return self.f_lineno
    try:
        return self.co.addr2line(self.f_lasti)
    except Exception:
        return None