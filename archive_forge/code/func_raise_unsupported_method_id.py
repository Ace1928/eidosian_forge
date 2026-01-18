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
def raise_unsupported_method_id(cls, coder):
    if coder['method'] == COMPRESSION_METHOD.P7Z_BCJ2:
        raise UnsupportedCompressionMethodError(coder['method'], 'BCJ2 filter is not supported by py7zr. Please consider to contribute to XZ/liblzma project and help Python core team implementing it. Or please use another tool to extract it.')
    if coder['method'] == COMPRESSION_METHOD.MISC_LZ4:
        raise UnsupportedCompressionMethodError(coder['method'], 'Archive is compressed by an unsupported algorythm LZ4.')
    raise UnsupportedCompressionMethodError(coder['method'], 'Archive is compressed by an unsupported compression algorythm.')