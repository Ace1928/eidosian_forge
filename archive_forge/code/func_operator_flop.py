from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
@classmethod
def operator_flop(cls, dO, q, k, v, b, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, logsumexp, output, dropout_p, rng_seed, rng_offset, custom_mask_type, scale) -> int:
    return cls.attn_operator_flop(q, k, v, seqstart_q=cu_seqlens_q, seqstart_k=cu_seqlens_k, causal=custom_mask_type > 0)