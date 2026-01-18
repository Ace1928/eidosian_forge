import locale
import os
import sys
from gitdb.utils.encoding import force_bytes, force_text  # noqa: F401  # @UnusedImport
from typing import (  # noqa: F401
def safe_encode(s: Optional[AnyStr]) -> Optional[bytes]:
    """Safely encode a binary string to Unicode."""
    if isinstance(s, str):
        return s.encode(defenc)
    elif isinstance(s, bytes):
        return s
    elif s is None:
        return None
    else:
        raise TypeError('Expected bytes or text, but got %r' % (s,))