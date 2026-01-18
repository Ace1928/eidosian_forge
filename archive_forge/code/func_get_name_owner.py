import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def get_name_owner(self, bus_name):
    """Return the unique connection name of the primary owner of the
        given name.

        :Raises `DBusException`: if the `bus_name` has no owner
        :Since: 0.81.0
        """
    validate_bus_name(bus_name, allow_unique=False)
    return self.call_blocking(BUS_DAEMON_NAME, BUS_DAEMON_PATH, BUS_DAEMON_IFACE, 'GetNameOwner', 's', (bus_name,))