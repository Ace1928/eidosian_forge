import gc
import math
from collections import namedtuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from xformers.benchmarks.utils import pretty_barplot
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import SparseCS, _matmul_with_mask
def add_kv(self, d, d_key, d_value, testcase):
    d_value = max(0, d_value)
    if d_key not in d:
        d[d_key] = {}
    d[d_key][testcase.name] = d_value