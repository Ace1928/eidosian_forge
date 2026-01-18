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
def paste_osx_pbcopy():
    with subprocess.Popen(['pbpaste', 'r'], stdout=subprocess.PIPE, close_fds=True) as p:
        stdout = p.communicate()[0]
    return stdout.decode(ENCODING)