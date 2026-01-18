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
def safe_tp_name(self):
    try:
        return self.field('self')['ob_type']['tp_name'].string()
    except (NullPyObjectPtr, RuntimeError, UnicodeDecodeError):
        return '<unknown tp_name>'