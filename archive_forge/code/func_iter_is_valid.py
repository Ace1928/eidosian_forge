import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def iter_is_valid(self, iter):
    """
        :Returns:
            True if the gtk.TreeIter specified by iter is valid for the custom tree model.
        """
    return iter.stamp == self.stamp