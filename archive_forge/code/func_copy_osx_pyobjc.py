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
def copy_osx_pyobjc(text):
    """Copy string argument to clipboard"""
    text = _stringifyText(text)
    newStr = Foundation.NSString.stringWithString_(text).nsstring()
    newData = newStr.dataUsingEncoding_(Foundation.NSUTF8StringEncoding)
    board = AppKit.NSPasteboard.generalPasteboard()
    board.declareTypes_owner_([AppKit.NSStringPboardType], None)
    board.setData_forType_(newData, AppKit.NSStringPboardType)