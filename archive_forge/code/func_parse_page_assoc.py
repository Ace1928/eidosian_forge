import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
def parse_page_assoc(self, segment: JBIG2Segment, page: int, field: bytes) -> int:
    if cast(JBIG2SegmentFlags, segment['flags'])['page_assoc_long']:
        field += self.stream.read(3)
        page = unpack_int('>L', field)
    return page