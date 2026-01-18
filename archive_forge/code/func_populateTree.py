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
def populateTree(self, root, examples):
    bold_font = None
    for key, val in examples.items():
        item = QtWidgets.QTreeWidgetItem([key])
        self.itemCache.append(item)
        if isinstance(val, OrderedDict):
            self.populateTree(item, val)
        elif isinstance(val, Namespace):
            item.file = val.filename
            if 'recommended' in val:
                if bold_font is None:
                    bold_font = item.font(0)
                    bold_font.setBold(True)
                item.setFont(0, bold_font)
        else:
            item.file = val
        root.addChild(item)