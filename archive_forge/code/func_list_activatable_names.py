import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def list_activatable_names(self):
    """Return a list of all names that can be activated on the bus.

        :Returns: a dbus.Array of dbus.UTF8String
        :Since: 0.81.0
        """
    return self.call_blocking(BUS_DAEMON_NAME, BUS_DAEMON_PATH, BUS_DAEMON_IFACE, 'ListActivatableNames', '', ())