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
def paste_xclip(primary=False):
    selection = DEFAULT_SELECTION
    if primary:
        selection = PRIMARY_SELECTION
    with subprocess.Popen(['xclip', '-selection', selection, '-o'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True) as p:
        stdout = p.communicate()[0]
    return stdout.decode(ENCODING)