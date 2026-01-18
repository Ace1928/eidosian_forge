from jeepney.bus_messages import message_bus
from jeepney.io.blocking import DBusConnection, Proxy, open_dbus_connection
from secretstorage.collection import Collection, create_collection, \
from secretstorage.item import Item
from secretstorage.exceptions import SecretStorageException, \
from secretstorage.util import add_match_rules
def dbus_init() -> DBusConnection:
    """Returns a new connection to the session bus, instance of
    jeepney's :class:`DBusConnection` class. This connection can
    then be passed to various SecretStorage functions, such as
    :func:`~secretstorage.collection.get_default_collection`.

    .. warning::
       The D-Bus socket will not be closed automatically. You can
       close it manually using the :meth:`DBusConnection.close` method,
       or you can use the :class:`contextlib.closing` context manager:

       .. code-block:: python

          from contextlib import closing
          with closing(dbus_init()) as conn:
              collection = secretstorage.get_default_collection(conn)
              items = collection.search_items({'application': 'myapp'})

       However, you will not be able to call any methods on the objects
       created within the context after you leave it.

    .. versionchanged:: 3.0
       Before the port to Jeepney, this function returned an
       instance of :class:`dbus.SessionBus` class.

    .. versionchanged:: 3.1
       This function no longer accepts any arguments.
    """
    try:
        connection = open_dbus_connection()
        add_match_rules(connection)
        return connection
    except KeyError as ex:
        reason = 'Environment variable {} is unset'.format(ex.args[0])
        raise SecretServiceNotAvailableException(reason) from ex
    except (ConnectionError, ValueError) as ex:
        raise SecretServiceNotAvailableException(str(ex)) from ex