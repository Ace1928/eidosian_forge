from __future__ import generators
from dbus.exceptions import DBusException
from _dbus_bindings import (
from dbus.bus import BusConnection
from dbus.lowlevel import SignalMessage
from dbus._compat import is_py2
def get_starter(private=False):
    """Static method that returns a connection to the starter bus.

        :Parameters:
            `private` : bool
                If true, do not return a shared connection.
        """
    return StarterBus(private=private)