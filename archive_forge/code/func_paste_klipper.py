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
def paste_klipper():
    with subprocess.Popen(['qdbus', 'org.kde.klipper', '/klipper', 'getClipboardContents'], stdout=subprocess.PIPE, close_fds=True) as p:
        stdout = p.communicate()[0]
    clipboardContents = stdout.decode(ENCODING)
    assert len(clipboardContents) > 0
    assert clipboardContents.endswith('\n')
    if clipboardContents.endswith('\n'):
        clipboardContents = clipboardContents[:-1]
    return clipboardContents