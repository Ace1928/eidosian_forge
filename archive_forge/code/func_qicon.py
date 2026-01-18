import os.path as op
import warnings
from ..Qt import QtGui, QtWidgets
@property
def qicon(self):
    if self._icon is None:
        self._build_qicon()
    return self._icon