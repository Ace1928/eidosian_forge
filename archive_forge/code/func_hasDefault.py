import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def hasDefault(self):
    """Returns True if this parameter has a default value."""
    return self.opts.get('default', None) is not None