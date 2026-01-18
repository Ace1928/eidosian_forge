import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def set_base_dir(self, base_dir):
    """ Set the base directory to be used for all relative filenames. """
    self._base_dir = base_dir
    self.icon_cache.set_base_dir(base_dir)