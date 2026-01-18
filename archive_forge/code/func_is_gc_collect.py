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
def is_gc_collect(self):
    """Is this frame "collect" within the garbage-collector?"""
    return self._gdbframe.name() == 'collect'