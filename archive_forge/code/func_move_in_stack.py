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
def move_in_stack(move_up):
    """Move up or down the stack (for the py-up/py-down command)"""
    frame = Frame.get_selected_python_frame()
    if not frame:
        print('Unable to locate python frame')
        return
    while frame:
        if move_up:
            iter_frame = frame.older()
        else:
            iter_frame = frame.newer()
        if not iter_frame:
            break
        if iter_frame.is_python_frame():
            if iter_frame.select():
                iter_frame.print_summary()
            return
        frame = iter_frame
    if move_up:
        print('Unable to find an older python frame')
    else:
        print('Unable to find a newer python frame')