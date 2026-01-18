import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
def encode_segment(self, segment: JBIG2Segment) -> bytes:
    data = b''
    for field_format, name in SEG_STRUCT:
        value = segment.get(name)
        encoder = getattr(self, 'encode_%s' % name, None)
        if callable(encoder):
            field = encoder(value, segment)
        else:
            field = pack(field_format, value)
        data += field
    return data