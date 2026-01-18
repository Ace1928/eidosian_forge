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
def copy_dev_clipboard(text):
    text = _stringifyText(text)
    if text == '':
        warnings.warn('Pyperclip cannot copy a blank string to the clipboard on Cygwin. This is effectively a no-op.', stacklevel=find_stack_level())
    if '\r' in text:
        warnings.warn('Pyperclip cannot handle \\r characters on Cygwin.', stacklevel=find_stack_level())
    with open('/dev/clipboard', 'w', encoding='utf-8') as fd:
        fd.write(text)