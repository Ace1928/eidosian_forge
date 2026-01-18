from _dbus_bindings import _Server
from dbus.connection import Connection
def connection_removed(self, conn):
    """Respond to the disconnection of a Connection.

        This base-class implementation just invokes the callbacks in
        the on_connection_removed attribute.

        :Parameters:
            `conn` : dbus.connection.Connection
                A D-Bus connection which has just become disconnected.

                The type of this parameter is whatever was passed
                to the Server constructor as the ``connection_class``.
        """
    if self.on_connection_removed:
        for cb in self.on_connection_removed:
            cb(conn)