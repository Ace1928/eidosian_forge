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
def get_inferior_unicode_postfix():
    try:
        gdb.parse_and_eval('PyUnicode_FromEncodedObject')
    except RuntimeError:
        try:
            gdb.parse_and_eval('PyUnicodeUCS2_FromEncodedObject')
        except RuntimeError:
            return 'UCS4'
        else:
            return 'UCS2'
    else:
        return ''