import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def efn(x):
    return x.encode(sys.getfilesystemencoding())