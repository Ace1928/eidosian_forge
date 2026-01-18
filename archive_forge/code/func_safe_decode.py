import locale
import os
import sys
from gitdb.utils.encoding import force_bytes, force_text  # noqa: F401  # @UnusedImport
from typing import (  # noqa: F401
def safe_decode(s: Union[AnyStr, None]) -> Optional[str]:
    """Safely decode a binary string to Unicode."""
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode(defenc, 'surrogateescape')
    elif s is None:
        return None
    else:
        raise TypeError('Expected bytes or text, but got %r' % (s,))