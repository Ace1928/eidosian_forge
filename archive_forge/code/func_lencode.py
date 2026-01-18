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
def lencode(x):
    """Encode an unsigned integer into a variable sized blob of bytes."""
    if x <= 250:
        return spack('<B', x)
    else:
        return spack('<BQ', 253, x)