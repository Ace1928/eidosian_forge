import contextlib
import ctypes
from ctypes import (
import os
import platform
from shutil import which as _executable_exists
import subprocess
import time
import warnings
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
def init_xsel_clipboard():
    DEFAULT_SELECTION = '-b'
    PRIMARY_SELECTION = '-p'

    def copy_xsel(text, primary=False):
        text = _stringifyText(text)
        selection_flag = DEFAULT_SELECTION
        if primary:
            selection_flag = PRIMARY_SELECTION
        with subprocess.Popen(['xsel', selection_flag, '-i'], stdin=subprocess.PIPE, close_fds=True) as p:
            p.communicate(input=text.encode(ENCODING))

    def paste_xsel(primary=False):
        selection_flag = DEFAULT_SELECTION
        if primary:
            selection_flag = PRIMARY_SELECTION
        with subprocess.Popen(['xsel', selection_flag, '-o'], stdout=subprocess.PIPE, close_fds=True) as p:
            stdout = p.communicate()[0]
        return stdout.decode(ENCODING)
    return (copy_xsel, paste_xsel)