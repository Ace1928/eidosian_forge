from typing import Any, Dict, List, Optional
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation, build_activation
from xformers.triton.fused_linear_layer import FusedLinear
def metric_conversion(ms):
    return flop * 1e-12 / (ms * 0.001)