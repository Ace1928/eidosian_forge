import functools
import io
import operator
import os
import struct
from binascii import unhexlify
from functools import reduce
from io import BytesIO
from operator import and_, or_
from struct import pack, unpack
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from py7zr.compressor import SevenZipCompressor, SevenZipDecompressor
from py7zr.exceptions import Bad7zFile
from py7zr.helpers import ArchiveTimestamp, calculate_crc32
from py7zr.properties import DEFAULT_FILTERS, MAGIC_7Z, PROPERTY
def read_real_uint64(file: BinaryIO) -> Tuple[int, bytes]:
    """read 8 bytes, return unpacked value as a little endian unsigned long long, and raw data."""
    res = file.read(8)
    a = unpack('<Q', res)[0]
    return (a, res)