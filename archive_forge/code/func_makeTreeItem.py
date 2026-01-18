import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def makeTreeItem(self, depth):
    """
        Return a TreeWidgetItem suitable for displaying/controlling the content of 
        this parameter. This is called automatically when a ParameterTree attempts
        to display this Parameter.
        Most subclasses will want to override this function.
        """
    itemClass = self.itemClass or _PARAM_ITEM_TYPES.get(self.opts['type'], ParameterItem)
    return itemClass(self, depth)