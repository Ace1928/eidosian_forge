from __future__ import absolute_import, division, print_function
import bz2
import hashlib
import logging
import os
import re
import struct
import sys
import types
import zlib
from io import BytesIO
def lendecode(f):
    """Decode an unsigned integer from a file."""
    n = strunpack('<B', f.read(1))[0]
    if n == 253:
        n = strunpack('<Q', f.read(8))[0]
    return n