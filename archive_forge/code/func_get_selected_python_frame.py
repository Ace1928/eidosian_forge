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
@classmethod
def get_selected_python_frame(cls):
    """Try to obtain the Frame for the python-related code in the selected
        frame, or None"""
    try:
        frame = cls.get_selected_frame()
    except gdb.error:
        return None
    while frame:
        if frame.is_python_frame():
            return frame
        frame = frame.older()
    return None