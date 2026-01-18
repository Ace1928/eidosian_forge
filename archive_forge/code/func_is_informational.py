from __future__ import annotations
from enum import IntEnum
@classmethod
def is_informational(cls, value: int) -> bool:
    """
        Returns `True` for 1xx status codes, `False` otherwise.
        """
    return 100 <= value <= 199