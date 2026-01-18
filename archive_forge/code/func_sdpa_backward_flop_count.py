import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
def sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape):
    total_flops = 0
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    _b4, _h4, _s4, _d4 = grad_out_shape
    assert b == _b2 == _b3 == _b4 and h == _h2 == _h3 == _h4 and (d_q == _d2)
    assert d_v == _d4 and s_k == _s3 and (s_q == _s4)
    total_flops = 0
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
    total_flops += bmm_flop((b * h, s_q, d_v), (b * h, d_v, s_k))
    total_flops += bmm_flop((b * h, s_k, s_q), (b * h, s_q, d_v))
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_q))
    total_flops += bmm_flop((b * h, d_q, s_q), (b * h, s_q, s_k))
    return total_flops