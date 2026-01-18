import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def invalidate_iters(self):
    """
        This method invalidates all TreeIter objects associated with this custom tree model
        and frees their locally pooled references.
        """
    self.stamp = random.randint(-2147483648, 2147483647)
    self._held_refs.clear()