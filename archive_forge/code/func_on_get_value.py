import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_get_value(self, node, column):
    """Overridable.

        :Parameters:
            node : object
            column : int
                Column index to get the value from.

        :Returns:
            The value of the column for the given node."""
    raise NotImplementedError