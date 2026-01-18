import struct
import time
from binascii import crc32
import aiokafka.codec as codecs
from aiokafka.codec import (
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
@classmethod
def record_overhead(cls, magic):
    try:
        return cls.RECORD_OVERHEAD[magic]
    except KeyError:
        raise ValueError('Unsupported magic: %d' % magic)