from _dbus_bindings import _Server
from dbus.connection import Connection
def connection_added(self, conn):
    """Respond to the creation of a new Connection.

        This base-class implementation just invokes the callbacks in
        the on_connection_added attribute.

        :Parameters:
            `conn` : dbus.connection.Connection
                A D-Bus connection which has just been added.

                The type of this parameter is whatever was passed
                to the Server constructor as the ``connection_class``.
        """
    if self.on_connection_added:
        for cb in self.on_connection_added:
            cb(conn)