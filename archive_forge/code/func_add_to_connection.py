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
def add_to_connection(self, connection, path):
    """Make this object accessible via the given D-Bus connection and
        object path.

        :Parameters:
            `connection` : dbus.connection.Connection
                Export the object on this connection. If the class attribute
                SUPPORTS_MULTIPLE_CONNECTIONS is False (default), this object
                can only be made available on one connection; if the class
                attribute is set True by a subclass, the object can be made
                available on more than one connection.

            `path` : dbus.ObjectPath or other str
                Place the object at this object path. If the class attribute
                SUPPORTS_MULTIPLE_OBJECT_PATHS is False (default), this object
                can only be made available at one object path; if the class
                attribute is set True by a subclass, the object can be made
                available with more than one object path.

        :Raises ValueError: if the object's class attributes do not allow the
            object to be exported in the desired way.
        :Since: 0.82.0
        """
    if path == LOCAL_PATH:
        raise ValueError('Objects may not be exported on the reserved path %s' % LOCAL_PATH)
    self._locations_lock.acquire()
    try:
        if self._connection is not None and self._connection is not connection and (not self.SUPPORTS_MULTIPLE_CONNECTIONS):
            raise ValueError('%r is already exported on connection %r' % (self, self._connection))
        if self._object_path is not None and (not self.SUPPORTS_MULTIPLE_OBJECT_PATHS) and (self._object_path != path):
            raise ValueError('%r is already exported at object path %s' % (self, self._object_path))
        connection._register_object_path(path, self._message_cb, self._unregister_cb, self._fallback)
        if self._connection is None:
            self._connection = connection
        elif self._connection is not connection:
            self._connection = _MANY
        if self._object_path is None:
            self._object_path = path
        elif self._object_path != path:
            self._object_path = _MANY
        self._locations.append((connection, path, self._fallback))
    finally:
        self._locations_lock.release()