from typing import Dict, Iterator, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX, SS_PATH
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, ItemNotFoundException, \
from secretstorage.item import Item
from secretstorage.util import DBusAddressWrapper, exec_prompt, \
def get_default_collection(connection: DBusConnection, session: Optional[Session]=None) -> Collection:
    """Returns the default collection. If it doesn't exist,
    creates it."""
    try:
        return Collection(connection)
    except ItemNotFoundException:
        return create_collection(connection, 'Default', 'default', session)