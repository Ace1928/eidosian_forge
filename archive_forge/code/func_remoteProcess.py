from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
import atexit
import enum
import mmap
import os
import sys
import tempfile
from .. import Qt
from .. import CONFIG_OPTIONS
from .. import multiprocess as mp
from .GraphicsView import GraphicsView
def remoteProcess(self):
    """Return the remote process handle. (see multiprocess.remoteproxy.RemoteEventHandler)"""
    return self._proc