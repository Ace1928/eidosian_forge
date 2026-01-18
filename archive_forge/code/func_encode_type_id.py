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
def encode_type_id(b, ext_id):
    """Encode the type identifier, with or without extension id."""
    if ext_id is not None:
        bb = ext_id.encode('UTF-8')
        return b.upper() + lencode(len(bb)) + bb
    else:
        return b