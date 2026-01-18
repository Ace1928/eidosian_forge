import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def set_user_data(self, iter, user_data):
    """Applies user_data and stamp to the given iter.

        If the models "leak_references" property is set, a reference to the
        user_data is stored with the model to ensure we don't run into bad
        memory problems with the TreeIter.
        """
    iter.user_data = id(user_data)
    if user_data is None:
        self.invalidate_iter(iter)
    else:
        iter.stamp = self.stamp
        if self.leak_references:
            self._held_refs[iter.user_data] = user_data