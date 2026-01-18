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
def get_truncated_repr(self, maxlen):
    """
        Get a repr-like string for the data, but truncate it at "maxlen" bytes
        (ending the object graph traversal as soon as you do)
        """
    out = TruncatedStringIO(maxlen)
    try:
        self.write_repr(out, set())
    except StringTruncated:
        return out.getvalue() + '...(truncated)'
    return out.getvalue()