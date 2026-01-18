import functools
import torch
from torch._inductor.compile_fx import fake_tensor_prop
from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
@functools.lru_cache(None)
def addmm_patterns_init():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    val = functools.partial(torch.empty, (10, 10), device=device, requires_grad=False)

    def check_concat_weights(match):
        weights = [match.kwargs['w1'], match.kwargs['w2']]
        if 'w3' in match.kwargs:
            weights.append(match.kwargs['w3'])
        return all((w.op == 'get_attr' and w.meta['val'].shape == weights[0].meta['val'].shape for w in weights))

    def matmul_fuse_pattern(inp, w1, w2, w3):
        return (inp @ w1, inp @ w2, inp @ w3)

    def matmul_replacement(inp, w1, w2, w3):
        cat_t = torch.cat((w1, w2, w3), dim=1)
        mm = inp @ cat_t
        return mm.chunk(3, dim=1)
    register_replacement(matmul_fuse_pattern, matmul_replacement, [val(), val(), val(), val()], fwd_only, pass_patterns[0], extra_check=check_concat_weights, exclusive_arg_names=('w1', 'w2', 'w3'))

    def matmul_fuse_pattern_two(inp, w1, w2):
        return (inp @ w1, inp @ w2)

    def matmul_replacement_two(inp, w1, w2):
        cat_t = torch.cat((w1, w2), dim=1)
        mm = inp @ cat_t
        return mm.chunk(2, dim=1)
    register_replacement(matmul_fuse_pattern_two, matmul_replacement_two, [val(), val(), val()], fwd_only, pass_patterns[0], extra_check=check_concat_weights, exclusive_arg_names=('w1', 'w2'))

    def addmm_fuse_pattern_second(inp, w1, w2, w3, b1, b2, b3):
        return (aten.addmm(b1, inp, w1), aten.addmm(b2, inp, w2), aten.addmm(b3, inp, w3))

    def addmm_fuse_replacement_second(inp, w1, w2, w3, b1, b2, b3):
        cat_w = torch.cat((w1, w2, w3), dim=1)
        cat_b = torch.cat((b1, b2, b3))
        return aten.addmm(cat_b, inp, cat_w).chunk(3, dim=1)
    register_replacement(addmm_fuse_pattern_second, addmm_fuse_replacement_second, [val() for _ in range(7)], fwd_only, pass_patterns[0], extra_check=check_concat_weights, exclusive_arg_names=('w1', 'w2', 'w3', 'b1', 'b2', 'b3'))