import math
import secrets
import uuid as _uu
from typing import List
from typing import Optional
def get_alphabet(self) -> str:
    """Return the current alphabet used for new UUIDs."""
    return ''.join(self._alphabet)