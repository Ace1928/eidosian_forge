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
def init_qt_clipboard():
    global QApplication
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError:
            from PyQt4.QtGui import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    def copy_qt(text):
        text = _stringifyText(text)
        cb = app.clipboard()
        cb.setText(text)

    def paste_qt() -> str:
        cb = app.clipboard()
        return str(cb.text())
    return (copy_qt, paste_qt)