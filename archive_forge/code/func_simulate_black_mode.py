import keyword
import os
import pkgutil
import re
import subprocess
import sys
from argparse import Namespace
from collections import OrderedDict
from functools import lru_cache
import pyqtgraph as pg
from pyqtgraph.Qt import QT_LIB, QtCore, QtGui, QtWidgets
import exampleLoaderTemplate_generic as ui_template
import utils
def simulate_black_mode(self):
    """
        used to simulate MacOS "black mode" on other platforms
        intended for debug only, as it manage only the QPlainTextEdit
        """
    c = QtGui.QColor('#171717')
    p = self.ui.codeView.palette()
    p.setColor(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Base, c)
    p.setColor(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Base, c)
    self.ui.codeView.setPalette(p)
    f = QtGui.QTextCharFormat()
    f.setForeground(QtGui.QColor('white'))
    self.ui.codeView.setCurrentCharFormat(f)
    app = QtWidgets.QApplication.instance()
    app.setProperty('darkMode', True)