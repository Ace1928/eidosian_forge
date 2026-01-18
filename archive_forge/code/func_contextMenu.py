import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def contextMenu(self, name):
    """"A context menu entry was clicked"""
    self.sigContextMenu.emit(self, name)