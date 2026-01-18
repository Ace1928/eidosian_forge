import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def encode_exception_table_varint(n: int) -> List[int]:
    """
    Similar to `encode_varint`, but the 6-bit chunks are ordered in reverse.
    """
    assert n >= 0
    b = [n & 63]
    n >>= 6
    while n > 0:
        b.append(n & 63)
        n >>= 6
    b = list(reversed(b))
    for i in range(len(b) - 1):
        b[i] |= 64
    return b