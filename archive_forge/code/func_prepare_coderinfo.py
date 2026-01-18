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
def prepare_coderinfo(self, filters):
    self.compressor = SevenZipCompressor(filters=filters, password=self.password)
    self.coders = self.compressor.coders
    assert len(self.coders) > 0
    self.solid = True
    self.digestdefined = False
    num_bindpairs = sum([c['numoutstreams'] for c in self.coders]) - 1
    self.bindpairs = [Bond(incoder=i + 1, outcoder=i) for i in range(num_bindpairs)]
    assert sum([c['numinstreams'] for c in self.coders]) == sum([c['numoutstreams'] for c in self.coders])