from enum import IntEnum
@classmethod
def fromOrdinal(cls, i: int):
    return cls._value2member_map_[i]