from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
@classmethod
def get_crc(cls, buf: Iterable[int]) -> bytes:
    new_crc = 15933696
    for byte in buf:
        for bit_count in range(8):
            new_crc >>= 1
            if byte & 1 << bit_count:
                new_crc |= 8388608
            if new_crc & 128:
                new_crc ^= 8652800
    for _bit_count in range(16):
        new_crc >>= 1
        if new_crc & 128:
            new_crc ^= 8652800
    return (~new_crc >> 8 & 65535).to_bytes(2, 'little')