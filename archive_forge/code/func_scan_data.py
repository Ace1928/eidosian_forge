import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def scan_data(self):
    offset = self._index % LEVELDBLOG_BLOCK_LEN
    space_left = LEVELDBLOG_BLOCK_LEN - offset
    if space_left < LEVELDBLOG_HEADER_LEN:
        pad_check = strtobytes('\x00' * space_left)
        pad = self._fp.read(space_left)
        assert pad == pad_check, 'invalid padding'
        self._index += space_left
    record = self.scan_record()
    if record is None:
        return None
    dtype, data = record
    if dtype == LEVELDBLOG_FULL:
        return data
    assert dtype == LEVELDBLOG_FIRST, f'expected record to be type {LEVELDBLOG_FIRST} but found {dtype}'
    while True:
        offset = self._index % LEVELDBLOG_BLOCK_LEN
        record = self.scan_record()
        if record is None:
            return None
        dtype, new_data = record
        if dtype == LEVELDBLOG_LAST:
            data += new_data
            break
        assert dtype == LEVELDBLOG_MIDDLE, f'expected record to be type {LEVELDBLOG_MIDDLE} but found {dtype}'
        data += new_data
    return data