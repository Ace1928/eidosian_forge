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
def applySearchHighlight(self, text):
    if not self.searchText:
        return
    expr = f'(?i){self.searchText}'
    palette: QtGui.QPalette = app.palette()
    color = palette.highlight().color()
    fgndColor = palette.color(palette.ColorGroup.Current, palette.ColorRole.Text).name()
    style = charFormat(fgndColor, background=color.name())
    for match in re.finditer(expr, text):
        start = match.start()
        length = match.end() - start
        self.setFormat(start, length, style)