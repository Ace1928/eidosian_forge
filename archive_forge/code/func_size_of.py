import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
@classmethod
def size_of(cls, key, value, headers):
    size = 0
    if key is None:
        size += 1
    else:
        key_len = len(key)
        size += size_of_varint(key_len) + key_len
    if value is None:
        size += 1
    else:
        value_len = len(value)
        size += size_of_varint(value_len) + value_len
    size += size_of_varint(len(headers))
    for h_key, h_value in headers:
        h_key_len = len(h_key.encode('utf-8'))
        size += size_of_varint(h_key_len) + h_key_len
        if h_value is None:
            size += 1
        else:
            h_value_len = len(h_value)
            size += size_of_varint(h_value_len) + h_value_len
    return size