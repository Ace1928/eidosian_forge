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
def older(self):
    older = self._gdbframe.older()
    if older:
        return Frame(older)
    else:
        return None