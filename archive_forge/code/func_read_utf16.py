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
def read_utf16(file: BinaryIO) -> str:
    """read a utf-16 string from file"""
    val = b''
    for _ in range(MAX_LENGTH):
        ch = file.read(2)
        if ch == unhexlify('0000'):
            break
        val += ch
    return val.decode('utf-16LE')