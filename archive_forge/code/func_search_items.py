from typing import Dict, Iterator, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX, SS_PATH
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, ItemNotFoundException, \
from secretstorage.item import Item
from secretstorage.util import DBusAddressWrapper, exec_prompt, \
def search_items(self, attributes: Dict[str, str]) -> Iterator[Item]:
    """Returns a generator of items with the given attributes.
        `attributes` should be a dictionary."""
    result, = self._collection.call('SearchItems', 'a{ss}', attributes)
    for item_path in result:
        yield Item(self.connection, item_path, self.session)