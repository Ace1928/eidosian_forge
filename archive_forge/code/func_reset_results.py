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
def reset_results(self):
    self.results = {}
    self.results['flops'] = {}
    self.results['time'] = {}
    self.results['memory'] = {}
    self.results['speedup'] = {}
    self.results['memory_savings'] = {}