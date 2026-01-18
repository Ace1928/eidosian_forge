import functools
import typing
from base64 import urlsafe_b64decode
from base64 import urlsafe_b64encode
from enum import Enum
@property
def sign_count(self) -> int:
    return self._sign_count