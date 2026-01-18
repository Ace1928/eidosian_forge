import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
@property
def needs_input(self):
    return self._needs_input