import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def add_toolitem(self, name, group, position, image_file, description, toggle):
    if toggle:
        button = Gtk.ToggleButton()
    else:
        button = Gtk.Button()
    button.set_label(name)
    button.add_css_class('flat')
    if image_file is not None:
        image = Gtk.Image.new_from_gicon(Gio.Icon.new_for_string(image_file))
        button.set_child(image)
        button.add_css_class('image-button')
    if position is None:
        position = -1
    self._add_button(button, group, position)
    signal = button.connect('clicked', self._call_tool, name)
    button.set_tooltip_text(description)
    self._toolitems.setdefault(name, [])
    self._toolitems[name].append((button, signal))