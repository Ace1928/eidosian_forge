import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
def set_producer_state(self, producer_id, producer_epoch, base_sequence):
    self._producer_id = producer_id
    self._producer_epoch = producer_epoch
    self._base_sequence = base_sequence