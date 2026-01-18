import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_iter_next(self, node):
    """Overridable.

        :Parameters:
            node : object
                Node at current level.

        :Returns:
            A python object (node) following the given node at the current level.
        """
    raise NotImplementedError