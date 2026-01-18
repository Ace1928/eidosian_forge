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
def delete_breakpoints(self, breakpoint_list):
    for bp in breakpoint_list:
        gdb.execute('delete %s' % bp)