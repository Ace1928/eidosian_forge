from typing import Any, Dict, List, Optional
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation, build_activation
from xformers.triton.fused_linear_layer import FusedLinear
def get_metrics_transform(activation: Optional[Activation], a: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], backward: bool):
    flop = a.shape[0] * a.shape[1] * w.shape[1] * (2 * a.shape[2] - 1)
    if activation is not None:
        flop += a.numel()
    if backward:
        flop *= 2
        flop += a.shape[0] * a.shape[1] * w.shape[1]
        flop += a.shape[0] * a.shape[1] * w.shape[1] * (2 * a.shape[2] - 1)
    if b is not None:
        flop += b.numel()

    def metric_conversion(ms):
        return flop * 1e-12 / (ms * 0.001)
    return metric_conversion