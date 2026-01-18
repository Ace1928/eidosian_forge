from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
def sample_inputs_multi_head_attention_forward(opinfo, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    if requires_grad:
        bsz = 2
        is_batcheds = (True,)
        use_separate_proj_weights = (False,)
        emb_sizes = (2,)
        src_lens = (XS,)
        tgt_lens = (XS,)
        heads = (2,)
        dropouts = (0.5,)
        mask_types = ('2d',)
    else:
        bsz = 2
        is_batcheds = (False, True)
        use_separate_proj_weights = (False, True)
        emb_sizes = (2, 4)
        src_lens = (XS,)
        tgt_lens = (XS, S)
        heads = (1, 2)
        dropouts = (0.0, 0.5)
        mask_types = (None, '2d', '3d')
    for is_batched, use_separate_proj_weight, mask_type, emb_size, src_len, tgt_len, num_heads, dropout_p in itertools.product(is_batcheds, use_separate_proj_weights, mask_types, emb_sizes, src_lens, tgt_lens, heads, dropouts):
        attn_mask = None
        if mask_type == '2d':
            attn_mask = make_input(src_len, tgt_len)
        elif mask_type == '3d':
            attn_mask = make_input((bsz if is_batched else 1) * num_heads, src_len, tgt_len)
        if is_batched:
            q = make_input(src_len, bsz, emb_size)
            k = make_input(tgt_len, bsz, emb_size)
            v = make_input(tgt_len, bsz, emb_size)
        else:
            q = make_input(src_len, emb_size)
            k = make_input(tgt_len, emb_size)
            v = make_input(tgt_len, emb_size)
        if use_separate_proj_weight:
            in_proj_weight = None
            q_proj_weight = make_input(emb_size, emb_size)
            k_proj_weight = make_input(emb_size, emb_size)
            v_proj_weight = make_input(emb_size, emb_size)
        else:
            in_proj_weight = make_input(emb_size * 3, emb_size)
            q_proj_weight = None
            k_proj_weight = None
            v_proj_weight = None
        bias_k = make_input(emb_size)
        bias_v = make_input(emb_size)
        in_proj_bias = make_input(emb_size * 3)
        out_proj_weight = make_input(emb_size, emb_size)
        out_proj_bias = make_input(emb_size)
        sample_args = (k, v, emb_size, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, False, dropout_p, out_proj_weight, out_proj_bias)
        sample_kwargs = {'q_proj_weight': q_proj_weight, 'k_proj_weight': k_proj_weight, 'v_proj_weight': v_proj_weight, 'attn_mask': attn_mask, 'training': True if dropout_p > 0.0 else False, 'use_separate_proj_weight': use_separate_proj_weight}
        yield SampleInput(q, args=sample_args, kwargs=sample_kwargs)