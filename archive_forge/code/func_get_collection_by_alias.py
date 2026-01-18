from typing import Dict, Iterator, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX, SS_PATH
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, ItemNotFoundException, \
from secretstorage.item import Item
from secretstorage.util import DBusAddressWrapper, exec_prompt, \
def get_collection_by_alias(connection: DBusConnection, alias: str) -> Collection:
    """Returns the collection with the given `alias`. If there is no
    such collection, raises
    :exc:`~secretstorage.exceptions.ItemNotFoundException`."""
    service = DBusAddressWrapper(SS_PATH, SERVICE_IFACE, connection)
    collection_path, = service.call('ReadAlias', 's', alias)
    if len(collection_path) <= 1:
        raise ItemNotFoundException('No collection with such alias.')
    return Collection(connection, collection_path)