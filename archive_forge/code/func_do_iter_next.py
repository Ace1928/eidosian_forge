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
def do_iter_next(self, iter):
    """Internal method."""
    if iter is None:
        next_data = self.on_iter_next(None)
    else:
        next_data = self.on_iter_next(self.get_user_data(iter))
    self.set_user_data(iter, next_data)
    return next_data is not None