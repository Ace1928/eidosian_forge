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
def is_waiting_for_gil(self):
    """Is this frame waiting on the GIL?"""
    name = self._gdbframe.name()
    if name:
        return name == 'take_gil'