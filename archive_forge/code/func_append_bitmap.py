import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def append_bitmap(self, bitmap):
    """Add a sub-bitmap to the multi-page bitmap."""
    with self._fi as lib:
        lib.FreeImage_AppendPage(self._bitmap, bitmap._bitmap)