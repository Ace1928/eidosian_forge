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
def install_breakpoints(self):
    all_locations = itertools.chain(self.lang_info.static_break_functions(), self.lang_info.runtime_break_functions())
    for location in all_locations:
        result = gdb.execute('break %s' % location, to_string=True)
        yield re.search('Breakpoint (\\d+)', result).group(1)