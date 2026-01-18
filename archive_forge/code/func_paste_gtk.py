import contextlib
import ctypes
import os
import platform
import subprocess
import sys
import time
import warnings
from ctypes import c_size_t, sizeof, c_wchar_p, get_errno, c_wchar
def paste_gtk():
    clipboardContents = gtk.Clipboard().wait_for_text()
    if clipboardContents is None:
        return ''
    else:
        return clipboardContents