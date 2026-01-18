import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
def encrypt_object(self, obj: PdfObject, idnum: int, generation: int) -> PdfObject:
    if not self._is_encryption_object(obj):
        return obj
    cf = self._make_crypt_filter(idnum, generation)
    return cf.encrypt_object(obj)