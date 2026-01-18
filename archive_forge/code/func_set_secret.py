from typing import Dict, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, PromptDismissedException
from secretstorage.util import DBusAddressWrapper, \
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
def set_secret(self, secret: bytes, content_type: str='text/plain') -> None:
    """Sets item secret to `secret`. If `content_type` is given,
        also sets the content type of the secret (``text/plain`` by
        default)."""
    self.ensure_not_locked()
    if not self.session:
        self.session = open_session(self.connection)
    _secret = format_secret(self.session, secret, content_type)
    self._item.call('SetSecret', '(oayays)', _secret)