import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
@handle_exception(False)
def do_iter_has_child(self, parent):
    """Internal method."""
    return self.on_iter_has_child(self.get_user_data(parent))