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
def copy_klipper(text):
    text = _stringifyText(text)
    with subprocess.Popen(['qdbus', 'org.kde.klipper', '/klipper', 'setClipboardContents', text.encode(ENCODING)], stdin=subprocess.PIPE, close_fds=True) as p:
        p.communicate(input=None)