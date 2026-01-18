from __future__ import annotations
from enum import IntEnum
@classmethod
def get_reason_phrase(cls, value: int) -> str:
    try:
        return codes(value).phrase
    except ValueError:
        return ''