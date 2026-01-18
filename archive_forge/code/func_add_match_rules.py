import os
from typing import Any, List, Tuple
from jeepney import (
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import DBUS_UNKNOWN_METHOD, DBUS_NO_SUCH_OBJECT, \
from secretstorage.dhcrypto import Session, int_to_bytes
from secretstorage.exceptions import ItemNotFoundException, \
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
def add_match_rules(connection: DBusConnection) -> None:
    """Adds match rules for the given connection.

    Currently it matches all messages from the Prompt interface, as the
    mock service (unlike GNOME Keyring) does not specify the signal
    destination.

    .. versionadded:: 3.1
    """
    rule = MatchRule(sender=BUS_NAME, interface=PROMPT_IFACE)
    dbus = DBusAddressWrapper(path='/org/freedesktop/DBus', interface='org.freedesktop.DBus', connection=connection)
    dbus.bus_name = 'org.freedesktop.DBus'
    dbus.call('AddMatch', 's', rule.serialise())