import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def install_child_property(container, flag, pspec):
    warnings.warn('install_child_property() is not supported', gi.PyGIDeprecationWarning, stacklevel=2)