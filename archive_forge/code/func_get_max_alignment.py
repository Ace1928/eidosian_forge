import functools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional
import sympy
import torch
from ...codecache import cache_dir
from ...config import cuda as inductor_cuda_config
from ...ir import Layout
from .cuda_env import get_cuda_arch, get_cuda_version
def get_max_alignment(inductor_layout: Layout) -> int:
    """
    Returns the max alignment (in terms of number of elements) for a given Inductor Layout.
    """
    dtype = inductor_layout.dtype
    size = inductor_layout.size
    offset = inductor_layout.offset

    def is_static_int(number):
        return isinstance(number, (int, sympy.Integer))
    if is_static_int(size[-1]) and is_static_int(offset):
        alignments = get_alignments(dtype)
        for alignment in alignments:
            if int(size[-1]) % alignment == 0 and int(offset) % alignment == 0:
                return alignment
    return 1