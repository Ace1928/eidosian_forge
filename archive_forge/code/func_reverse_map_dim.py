import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def reverse_map_dim(dim_order, d):
    if dim_order in (DimOrder.PRESUMED_CONTIGUOUS, DimOrder.SCALAR_OR_VECTOR):
        return d
    assert dim_order is DimOrder.CHANNELS_LAST
    return [0, 2, 3, 1][d]