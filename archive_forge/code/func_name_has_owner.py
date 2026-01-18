import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def name_has_owner(self, bus_name):
    """Return True iff the given bus name has an owner on this bus.

        :Parameters:
            `bus_name` : str
                The bus name to look up
        :Returns: a `bool`
        """
    return bool(self.call_blocking(BUS_DAEMON_NAME, BUS_DAEMON_PATH, BUS_DAEMON_IFACE, 'NameHasOwner', 's', (bus_name,)))