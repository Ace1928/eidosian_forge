import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def childPath(self, child):
    """
        Return the path of parameter names from self to child.
        If child is not a (grand)child of self, return None.
        """
    path = []
    while child is not self:
        path.insert(0, child.name())
        child = child.parent()
        if child is None:
            return None
    return path