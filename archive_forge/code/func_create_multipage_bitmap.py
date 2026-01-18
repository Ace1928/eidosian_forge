import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def create_multipage_bitmap(self, filename, ftype, flags=0):
    """create_multipage_bitmap(filename, ftype, flags=0)
        Create a wrapped multipage bitmap object.
        """
    return FIMultipageBitmap(self, filename, ftype, flags)