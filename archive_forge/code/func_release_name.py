import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def release_name(self, name):
    """Release a bus name.

        :Parameters:
            `name` : str
                The well-known name to be released
        :Returns: `RELEASE_NAME_REPLY_RELEASED`,
            `RELEASE_NAME_REPLY_NON_EXISTENT`
            or `RELEASE_NAME_REPLY_NOT_OWNER`
        :Raises `DBusException`: if the bus daemon cannot be contacted or
            returns an error.
        """
    validate_bus_name(name, allow_unique=False)
    return self.call_blocking(BUS_DAEMON_NAME, BUS_DAEMON_PATH, BUS_DAEMON_IFACE, 'ReleaseName', 's', (name,))