import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
@property
def compression_type(self):
    return self.attributes & self.CODEC_MASK