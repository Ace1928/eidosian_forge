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
def get_methods_names(coders_lists: List[List[dict]]) -> List[str]:
    methods_namelist = ['LZMA2', 'LZMA', 'BZip2', 'DEFLATE', 'DEFLATE64', 'delta', 'COPY', 'PPMd', 'ZStandard', 'LZ4*', 'BCJ2*', 'BCJ', 'ARM', 'ARMT', 'IA64', 'PPC', 'SPARC', '7zAES']
    unsupported_methods = {COMPRESSION_METHOD.P7Z_BCJ2: 'BCJ2*', COMPRESSION_METHOD.MISC_LZ4: 'LZ4*'}
    methods_names = []
    for coders in coders_lists:
        for coder in coders:
            for m in SupportedMethods.methods:
                if coder['method'] == m['id']:
                    methods_names.append(m['name'])
            if coder['method'] in unsupported_methods:
                methods_names.append(unsupported_methods[coder['method']])
    return list(filter(lambda x: x in methods_names, methods_namelist))