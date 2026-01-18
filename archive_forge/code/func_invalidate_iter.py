import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def invalidate_iter(self, iter):
    """Clear user data and its reference from the iter and this model."""
    iter.stamp = 0
    if iter.user_data:
        if iter.user_data in self._held_refs:
            del self._held_refs[iter.user_data]
        iter.user_data = None