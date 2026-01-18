import itertools
import random
from functools import partial
import torch
from torch.utils import benchmark
import xformers.ops
import xformers.ops.fmha as fmha
from xformers.attn_bias_utils import create_attn_bias
from xformers.benchmarks.utils import benchmark_main_helper
def mem_eff_attention_bw(shape, num_threads: int, attn_bias_cfg, dropout_p, dtype):
    B, M, H, K = shape
    qkv, q, k, v = create_tensors(shape, dtype, requires_grad=True)
    attn_bias_type, attn_bias_requires_grad = attn_bias_cfg
    dtype_str = {torch.bfloat16: 'b16', torch.half: 'f16', torch.float: 'f32'}[dtype]
    sub_label = f'{dtype_str} {B}-{M}-{H}-{K}, p={dropout_p}, BiasT={attn_bias_type.__name__}, BiasGrad={attn_bias_requires_grad}'
    has_run = False
    for fw_op, bw_op in OPS:
        bias = create_attn_bias(attn_bias_type, batch_size=B, num_heads=H, num_heads_groups=1, q_len=M, kv_len=M, dtype=dtype, device=device, requires_grad=attn_bias_requires_grad, fmt='BMHK', op=bw_op)
        inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=bias, p=dropout_p)
        if not fw_op.supports(inp) or not bw_op.supports(inp):
            continue
        has_run = True
        out = xformers.ops.memory_efficient_attention(inp.query, inp.key, inp.value, inp.attn_bias, inp.p, op=(fw_op, bw_op))
        grad_benchmark = torch.ones_like(q)
        yield benchmark.Timer(stmt='out.backward(grad, retain_graph=True)', globals={'out': out, 'grad': grad_benchmark}, label=f'attention backward (attn_bias={attn_bias_type})', description=bw_op.NAME, sub_label=sub_label, num_threads=num_threads)
        del out
    if not has_run:
        return
    yield benchmark.Timer(stmt='out.backward(grad, retain_graph=True)', globals={'out': ref_attention(q, k, v, inp.attn_bias, dropout_p), 'grad': grad_benchmark}, label=f'attention backward (attn_bias={attn_bias_type})', description='vanilla', sub_label=sub_label, num_threads=num_threads)