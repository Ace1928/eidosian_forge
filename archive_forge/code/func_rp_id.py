import functools
import typing
from base64 import urlsafe_b64decode
from base64 import urlsafe_b64encode
from enum import Enum
@property
def rp_id(self) -> str:
    return self._rp_id