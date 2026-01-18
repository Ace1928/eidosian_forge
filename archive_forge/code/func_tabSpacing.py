import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def tabSpacing(self, widget, prop):
    prop_value = self.convert(prop)
    if prop_value is not None:
        self.delayed_props.append((widget, True, 'setSpacing', prop_value))