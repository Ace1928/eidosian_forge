import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def key_release_event(self, controller, keyval, keycode, state):
    KeyEvent('key_release_event', self, self._get_key(keyval, keycode, state), *self._mpl_coords())._process()
    return True