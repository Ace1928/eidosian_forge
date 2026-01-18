import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_iter_nth_child(self, parent, n):
    """Overridable.

        :Parameters:
            parent : object
            n : int
                Index of child within parent.

        :Returns:
            The child for the given parent index starting at 0. If parent None,
            return the top level node corresponding to "n".
            If "n" is larger then available nodes, return None.
        """
    raise NotImplementedError