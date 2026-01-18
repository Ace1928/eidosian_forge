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
def exc_info(self, frame):
    try:
        tstate = frame.read_var('tstate').dereference()
        if gdb.parse_and_eval('tstate->frame == f'):
            if sys.version_info >= (3, 12, 0, 'alpha', 6):
                inf_type = inf_value = tstate['current_exception']
            else:
                inf_type = tstate['curexc_type']
                inf_value = tstate['curexc_value']
            if inf_type:
                return 'An exception was raised: %s' % (inf_value,)
    except (ValueError, RuntimeError):
        pass