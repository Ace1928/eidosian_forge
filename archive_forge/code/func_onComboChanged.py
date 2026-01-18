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
def onComboChanged(searchType):
    if self.curListener is not None:
        self.curListener.disconnect()
    self.curListener = textFil.textChanged
    self.ui.exampleFilter.setStyleSheet('')
    if searchType == 'Content Search':
        self.curListener.connect(self.filterByContent)
    else:
        self.hl.searchText = None
        self.curListener.connect(self.filterByTitle)
    self.curListener.emit(textFil.text())