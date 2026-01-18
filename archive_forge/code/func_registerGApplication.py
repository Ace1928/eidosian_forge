from typing import Union
from gi.repository import GLib
from twisted.internet import _glibbase
from twisted.internet.error import ReactorAlreadyRunning
from twisted.python import runtime
def registerGApplication(self, app):
    """
        Register a C{Gio.Application} or C{Gtk.Application}, whose main loop
        will be used instead of the default one.
        """
    raise NotImplementedError('GApplication is not currently supported on Windows.')