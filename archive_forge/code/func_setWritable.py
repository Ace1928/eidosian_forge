import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def setWritable(self, writable=True):
    """Set whether this Parameter should be editable by the user. (This is 
        exactly the opposite of setReadonly)."""
    self.setOpts(readonly=not writable)