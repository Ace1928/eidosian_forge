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
def check_all(self, tests, a, b):
    ref_test = tests[0]
    ref_out = ref_test.prepare_callable(a, b, ref_test.mask, ref_test.config)()
    res_test = tests[1]
    res_out = res_test.prepare_callable(a, b, res_test.mask, res_test.config)()
    self.check_accuracy(ref_out, res_out, ref_test.mask, ref_test.config)