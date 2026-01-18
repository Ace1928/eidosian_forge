import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def compile_mmh():
    if not has_cupy_gpu:
        return None
    return cupy.RawKernel((PWD / '_murmur3.cu').read_text(encoding='utf8'), 'hash_data')