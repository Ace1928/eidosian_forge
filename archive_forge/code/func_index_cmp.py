import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
@functools.cmp_to_key
def index_cmp(a, b):
    if sizes[a] == 1 or sizes[b] == 1:
        return cmp(sizes[a] == 1, sizes[b] == 1)
    stride_len_a = [sl[a] for sl in stride_lengths]
    stride_len_b = [sl[b] for sl in stride_lengths]
    a_first = sum((sl_b == 0 or sl_a < sl_b for sl_a, sl_b in zip(stride_len_a, stride_len_b)))
    b_first = sum((sl_a == 0 or sl_b < sl_a for sl_a, sl_b in zip(stride_len_a, stride_len_b)))
    if a_first > b_first:
        return -1
    if b_first > a_first:
        return 1
    return cmp(b, a)