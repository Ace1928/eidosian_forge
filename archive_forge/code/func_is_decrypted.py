import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
def is_decrypted(self) -> bool:
    return self._password_type != PasswordType.NOT_DECRYPTED