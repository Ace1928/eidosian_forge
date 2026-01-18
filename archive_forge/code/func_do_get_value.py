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
def do_get_value(self, iter, column):
    """Internal method."""
    return self.on_get_value(self.get_user_data(iter), column)