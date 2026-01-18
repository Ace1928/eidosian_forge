import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def create_bitmap(self, filename, ftype, flags=0):
    """create_bitmap(filename, ftype, flags=0)
        Create a wrapped bitmap object.
        """
    return FIBitmap(self, filename, ftype, flags)