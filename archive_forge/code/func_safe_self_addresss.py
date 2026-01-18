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
def safe_self_addresss(self):
    try:
        address = long(self.field('self'))
        return '%#x' % address
    except (NullPyObjectPtr, RuntimeError):
        return '<failed to get self address>'