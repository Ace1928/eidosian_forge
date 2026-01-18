import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def isWrapping(self, widget, prop):
    widget.setWrapping(self.convert(prop))