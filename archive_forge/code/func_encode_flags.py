import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
def encode_flags(self, value: JBIG2SegmentFlags, segment: JBIG2Segment) -> bytes:
    flags = 0
    if value.get('deferred'):
        flags |= HEADER_FLAG_DEFERRED
    if 'page_assoc_long' in value:
        flags |= HEADER_FLAG_PAGE_ASSOC_LONG if value['page_assoc_long'] else flags
    else:
        flags |= HEADER_FLAG_PAGE_ASSOC_LONG if cast(int, segment.get('page', 0)) > 255 else flags
    flags |= mask_value(SEG_TYPE_MASK, value['type'])
    return pack('>B', flags)