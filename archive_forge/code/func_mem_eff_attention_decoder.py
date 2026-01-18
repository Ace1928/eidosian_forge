import itertools
from functools import partial
import torch
from torch.utils import benchmark
from utils import benchmark_main_helper
import xformers.ops
import xformers.ops.fmha as fmha
def mem_eff_attention_decoder(kv_shape, n_heads: int, num_threads: int, multiquery: bool):
    n_keys, padding, B = kv_shape
    torch.manual_seed(42)
    k_seqlen = torch.randint(1, n_keys + 1, (B,)).tolist()
    K = 128
    q = torch.rand(1, B, n_heads, K, device=device, dtype=torch.bfloat16)
    if multiquery:
        k = torch.rand(1, B * padding, 1, K, device=device, dtype=torch.bfloat16).expand(1, B * padding, n_heads, K)
        v = torch.rand(1, B * padding, 1, K, device=device, dtype=torch.bfloat16).expand(1, B * padding, n_heads, K)
    else:
        k = torch.rand(1, B * padding, n_heads, K, device=device, dtype=torch.bfloat16)
        v = torch.rand(1, B * padding, n_heads, K, device=device, dtype=torch.bfloat16)
    bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(q_seqlen=[1] * B, kv_seqlen=k_seqlen, kv_padding=padding)
    sub_label = f'{B}batch-{k_seqlen[0]}keys-{n_heads}heads'
    if multiquery:
        sub_label += '-mq'
    has_run = False
    for fw_op in OPS:
        inp = fmha.Inputs(q, k, v, attn_bias=bias)
        if not fw_op.supports(inp):
            continue
        fn = partial(xformers.ops.memory_efficient_attention_forward, op=fw_op)
        yield benchmark.Timer(stmt='fn(q, k, v, attn_bias)', globals={'q': q, 'k': k, 'v': v, 'attn_bias': bias, 'fn': fn}, label='attention', description=fw_op.NAME, sub_label=sub_label, num_threads=num_threads)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(q, k, v, bias)
        yield benchmark.Timer(stmt='graph.replay()', globals={'graph': graph}, label='cuda graphed attention', description=fw_op.NAME, sub_label=sub_label, num_threads=num_threads)
        has_run = True
    if not has_run:
        return
    RUN_BASELINES = False
    if RUN_BASELINES:
        yield benchmark.Timer(stmt='fn(q, k, v, attn_bias)', globals={'q': q, 'k': k, 'v': v, 'attn_bias': bias, 'fn': ref_attention}, label='attention', description='eager', sub_label=sub_label, num_threads=num_threads)