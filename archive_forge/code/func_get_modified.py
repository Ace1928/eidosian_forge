from typing import Dict, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, PromptDismissedException
from secretstorage.util import DBusAddressWrapper, \
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
def get_modified(self) -> int:
    """Returns UNIX timestamp (integer) representing the time
        when the item was last modified."""
    modified = self._item.get_property('Modified')
    assert isinstance(modified, int)
    return modified