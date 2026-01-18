import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def int_list(prop):
    return [int(child.text) for child in prop]