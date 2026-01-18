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
def print_traceback(self):
    if self.is_evalframe():
        pyop = self.get_pyop()
        if pyop:
            pyop.print_traceback()
            if not pyop.is_optimized_out():
                line = pyop.current_line()
                if line is not None:
                    sys.stdout.write('    %s\n' % line.strip())
        else:
            sys.stdout.write('  (unable to read python frame information)\n')
    else:
        info = self.is_other_python_frame()
        if info:
            sys.stdout.write('  %s\n' % info)
        else:
            sys.stdout.write('  (not a python frame)\n')