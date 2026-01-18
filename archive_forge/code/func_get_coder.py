import bz2
import lzma
import struct
import sys
import zlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import bcj
import inflate64
import pyppmd
import pyzstd
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from py7zr.exceptions import PasswordRequired, UnsupportedCompressionMethodError
from py7zr.helpers import Buffer, calculate_crc32, calculate_key
from py7zr.properties import (
@classmethod
def get_coder(cls, filter) -> Dict[str, Any]:
    method = cls.get_method_id(filter['id'])
    if filter['id'] in [lzma.FILTER_LZMA1, lzma.FILTER_LZMA2, lzma.FILTER_DELTA]:
        properties: Optional[bytes] = lzma._encode_filter_properties(filter)
    else:
        properties = None
    return {'method': method, 'properties': properties, 'numinstreams': 1, 'numoutstreams': 1}