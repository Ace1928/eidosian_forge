import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def iter_depth_first(self):
    """Depth-first iteration of the entire TreeModel yielding the python nodes."""
    stack = collections.deque([None])
    while stack:
        it = stack.popleft()
        if it is not None:
            yield self.get_user_data(it)
        children = [self.iter_nth_child(it, i) for i in range(self.iter_n_children(it))]
        stack.extendleft(reversed(children))