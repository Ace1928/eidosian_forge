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
def remoteSceneChanged(self, data):
    w, h, size = data
    if self.shm is None or self.shm.size != size:
        if self.shm is not None:
            self.shm.close()
        self.shm = mmap.mmap(self.shmFile.fileno(), size, access=mmap.ACCESS_READ)
    self._img = QtGui.QImage(self.shm, w, h, QtGui.QImage.Format.Format_RGB32).copy()
    self.update()