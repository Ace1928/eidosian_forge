import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_get_iter(self, path):
    """Overridable.

        :Returns:
            A python object (node) for the given TreePath.
        """
    raise NotImplementedError