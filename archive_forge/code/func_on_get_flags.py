import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_get_flags(self):
    """Overridable.

        :Returns Gtk.TreeModelFlags:
            The flags for this model. See: Gtk.TreeModelFlags
        """
    raise NotImplementedError