import sys
import logging
import threading
import traceback
import _dbus_bindings
from dbus import (
from dbus.decorators import method, signal
from dbus.exceptions import (
from dbus.lowlevel import ErrorMessage, MethodReturnMessage, MethodCallMessage
from dbus.proxies import LOCAL_PATH
from dbus._compat import is_py2
def remove_from_connection(self, connection=None, path=None):
    """Make this object inaccessible via the given D-Bus connection
        and object path. If no connection or path is specified,
        the object ceases to be accessible via any connection or path.

        :Parameters:
            `connection` : dbus.connection.Connection or None
                Only remove the object from this Connection. If None,
                remove from all Connections on which it's exported.
            `path` : dbus.ObjectPath or other str, or None
                Only remove the object from this object path. If None,
                remove from all object paths.
        :Raises LookupError:
            if the object was not exported on the requested connection
            or path, or (if both are None) was not exported at all.
        :Since: 0.81.1
        """
    self._locations_lock.acquire()
    try:
        if self._object_path is None or self._connection is None:
            raise LookupError('%r is not exported' % self)
        if connection is not None or path is not None:
            dropped = []
            for location in self._locations:
                if (connection is None or location[0] is connection) and (path is None or location[1] == path):
                    dropped.append(location)
        else:
            dropped = self._locations
            self._locations = []
        if not dropped:
            raise LookupError('%r is not exported at a location matching (%r,%r)' % (self, connection, path))
        for location in dropped:
            try:
                location[0]._unregister_object_path(location[1])
            except LookupError:
                pass
            if self._locations:
                try:
                    self._locations.remove(location)
                except ValueError:
                    pass
    finally:
        self._locations_lock.release()