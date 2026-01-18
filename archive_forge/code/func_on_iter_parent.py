import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_iter_parent(self, child):
    """Overridable.

        :Returns:
            The parent node of child or None if child is a top level node."""
    raise NotImplementedError