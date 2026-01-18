import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def remove_toolitem(self, name):
    if name not in self._toolitems:
        self.toolmanager.message_event(f'{name} not in toolbar', self)
        return
    for group in self._groups:
        for toolitem, _signal in self._toolitems[name]:
            if toolitem in self._groups[group]:
                self._groups[group].remove(toolitem)
    del self._toolitems[name]