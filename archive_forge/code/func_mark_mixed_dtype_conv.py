import functools
import itertools
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern
def mark_mixed_dtype_conv(conv):
    conv_dtype = conv.meta['val'].dtype
    if conv_dtype not in (torch.float16, torch.bfloat16):
        return
    if not len(conv.users) == 1:
        return
    conv_user = next(iter(conv.users.keys()))
    if not isinstance(conv_user.meta['val'], torch.Tensor):
        return
    if not conv_user.meta['val'].dtype == torch.float32:
        return
    while conv_user.target in _binary_ops:
        if not len(conv_user.users) == 1:
            return
        conv_user = next(iter(conv_user.users.keys()))
    if not (conv_user.target == prims.convert_element_type.default and conv_user.args[1] == conv_dtype):
        return
    conv.meta['_allow_conv_mixed_dtype_folding'] = conv_dtype