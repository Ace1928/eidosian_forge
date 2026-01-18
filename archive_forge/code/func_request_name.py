import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def request_name(self, name, flags=0):
    """Request a bus name.

        :Parameters:
            `name` : str
                The well-known name to be requested
            `flags` : dbus.UInt32
                A bitwise-OR of 0 or more of the flags
                `NAME_FLAG_ALLOW_REPLACEMENT`,
                `NAME_FLAG_REPLACE_EXISTING`
                and `NAME_FLAG_DO_NOT_QUEUE`
        :Returns: `REQUEST_NAME_REPLY_PRIMARY_OWNER`,
            `REQUEST_NAME_REPLY_IN_QUEUE`,
            `REQUEST_NAME_REPLY_EXISTS` or
            `REQUEST_NAME_REPLY_ALREADY_OWNER`
        :Raises `DBusException`: if the bus daemon cannot be contacted or
            returns an error.
        """
    validate_bus_name(name, allow_unique=False)
    return self.call_blocking(BUS_DAEMON_NAME, BUS_DAEMON_PATH, BUS_DAEMON_IFACE, 'RequestName', 'su', (name, flags))