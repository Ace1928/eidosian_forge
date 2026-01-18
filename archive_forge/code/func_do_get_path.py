import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
@handle_exception(None)
def do_get_path(self, iter):
    """Internal method."""
    path = self.on_get_path(self.get_user_data(iter))
    if path is None:
        return None
    else:
        return Gtk.TreePath(path)