from typing import Dict, Iterator, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX, SS_PATH
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, ItemNotFoundException, \
from secretstorage.item import Item
from secretstorage.util import DBusAddressWrapper, exec_prompt, \
def get_all_collections(connection: DBusConnection) -> Iterator[Collection]:
    """Returns a generator of all available collections."""
    service = DBusAddressWrapper(SS_PATH, SERVICE_IFACE, connection)
    for collection_path in service.get_property('Collections'):
        yield Collection(connection, collection_path)