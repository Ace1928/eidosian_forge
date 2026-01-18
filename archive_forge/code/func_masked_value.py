import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
def masked_value(mask: int, value: int) -> int:
    for bit_pos in range(0, 31):
        if bit_set(bit_pos, mask):
            return (value & mask) >> bit_pos
    raise Exception('Invalid mask or value')