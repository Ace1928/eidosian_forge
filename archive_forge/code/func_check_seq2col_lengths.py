import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def check_seq2col_lengths(lengths, B):
    if lengths is None:
        lengths = cupy.array([B], dtype='int32')
    else:
        _check_lengths(lengths, B)
    return lengths