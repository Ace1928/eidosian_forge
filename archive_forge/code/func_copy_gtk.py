import contextlib
import ctypes
import os
import platform
import subprocess
import sys
import time
import warnings
from ctypes import c_size_t, sizeof, c_wchar_p, get_errno, c_wchar
def copy_gtk(text):
    global cb
    text = _stringifyText(text)
    cb = gtk.Clipboard()
    cb.set_text(text)
    cb.store()