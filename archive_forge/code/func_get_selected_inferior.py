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
def get_selected_inferior():
    """
    Return the selected inferior in gdb.
    """
    return gdb.inferiors()[0]
    selected_thread = gdb.selected_thread()
    for inferior in gdb.inferiors():
        for thread in inferior.threads():
            if thread == selected_thread:
                return inferior