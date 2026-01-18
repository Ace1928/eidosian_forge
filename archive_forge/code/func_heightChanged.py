import contextlib
import re
import xml.dom.minidom as xml
import numpy as np
from .. import debug
from .. import functions as fn
from ..parametertree import Parameter
from ..Qt import QtCore, QtGui, QtSvg, QtWidgets
from .Exporter import Exporter
def heightChanged(self):
    sr = self.getSourceRect()
    ar = sr.width() / sr.height()
    self.params.param('width').setValue(self.params['height'] * ar, blockSignal=self.widthChanged)