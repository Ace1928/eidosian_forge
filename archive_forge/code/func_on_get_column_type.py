import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_get_column_type(self, index):
    """Overridable.

        :Returns:
            The column type for the given index.
        """
    raise NotImplementedError