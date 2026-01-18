import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def get_output_log(self):
    """Return a list of the last 256 output messages
        (warnings and errors) produced by the FreeImage library.
        """
    return [m for m in self._messages]