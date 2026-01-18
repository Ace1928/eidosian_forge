import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
def get_pts(time_base):
    start_offset = start_pts
    end_offset = end_pts
    if pts_unit == 'sec':
        start_offset = int(math.floor(start_pts * (1 / time_base)))
        if end_offset != float('inf'):
            end_offset = int(math.ceil(end_pts * (1 / time_base)))
    if end_offset == float('inf'):
        end_offset = -1
    return (start_offset, end_offset)