import base64
import binascii
import re
from typing import Union
def is_ssh_key(key: bytes) -> bool:
    if any((string_value in key for string_value in _SSH_KEY_FORMATS)):
        return True
    ssh_pubkey_match = _SSH_PUBKEY_RC.match(key)
    if ssh_pubkey_match:
        key_type = ssh_pubkey_match.group(1)
        if _CERT_SUFFIX == key_type[-len(_CERT_SUFFIX):]:
            return True
    return False