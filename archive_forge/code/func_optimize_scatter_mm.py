import inspect
import re
import warnings
from typing import Any, Dict
import torch
from torch.testing import make_tensor
def optimize_scatter_mm(m, k, n, bm, bk, dtype=torch.float16, device='cuda', sparsity=0.5, force=False):
    import triton
    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data
    key = (m, k, n, bm, bk)
    version = (0, dtype, sparsity)
    device_name = torch.cuda.get_device_name()
    reference_meta = dict(GROUP_SIZE=1, TILE_M=16, TILE_N=16, SPLIT_N=n // 16, num_stages=1, num_warps=1)
    initial_meta = get_meta('scatter_mm', key, device_name=device_name, version=version, exact=True)
    if initial_meta is None:
        initial_meta = get_meta('bsr_dense_mm', key, device_name=device_name, version=(0, dtype, 0.5), exact=True)
        if initial_meta is None:
            initial_meta = reference_meta
    elif not force:
        return
    print(f'm, k, n, bm, bk, initial_meta, reference_meta={(m, k, n, bm, bk, initial_meta, reference_meta)!r}')
    torch.manual_seed(0)
    bsr = create_blocked_tensor(0, m, k, (bm, bk), sparsity, dtype, device).to_sparse_bsr((bm, bk))
    dense = make_tensor(k, n, dtype=dtype, device=device)

    def bench(meta, bsr=bsr, dense=dense):
        indices_data = bsr_scatter_mm_indices_data(bsr, dense, indices_format='bsr_strided_mm_compressed', **meta)

        def test_func():
            return bsr_scatter_mm(bsr, dense, indices_data=indices_data)
        ms_min = triton.testing.do_bench(test_func, warmup=500, rep=100, fast_flush=False)
        return ms_min

    def step_meta_parameter(name, value, direction, meta, m=m, n=n, k=k, bm=bm, bk=bk):
        is_log = name in {'SPLIT_N', 'TILE_M', 'TILE_N', 'num_warps'}
        min_value = dict(SPLIT_N=1, TILE_M=16, TILE_N=16, num_warps=1, num_stages=1, GROUP_SIZE=1)[name]
        max_value = dict(SPLIT_N=n // meta['TILE_N'], TILE_M=bm, TILE_N=n // meta['SPLIT_N']).get(name)
        value_step = dict(SPLIT_N=2, TILE_M=2, TILE_N=2, num_warps=2, num_stages=1, GROUP_SIZE=1)[name]
        if is_log:
            next_value = value * value_step ** direction if direction > 0 else value // value_step ** abs(direction)
        else:
            next_value = value + value_step * direction
        if min_value is not None:
            next_value = max(next_value, min_value)
        if max_value is not None:
            next_value = min(next_value, max_value)
        if name == 'SPLIT_N' and n % next_value != 0:
            return value
        if (dtype, name, next_value, m, n, k, bm, bk) in {(torch.float32, 'num_warps', 32, 256, 256, 256, 16, 16), (torch.float32, 'num_warps', 16, 256, 256, 256, 32, 32), (torch.float32, 'num_warps', 16, 256, 256, 256, 64, 64), (torch.float32, 'num_warps', 16, 256, 256, 256, 128, 128), (torch.float32, 'num_warps', 16, 512, 512, 256, 128, 128)} and re.match('NVIDIA A100[^\\d]', device_name) is not None:
            return value
        return next_value
    meta, speedup, timing, sensitivity_message = minimize(bench, initial_meta, reference_meta, step_meta_parameter)
    if initial_meta is not reference_meta and initial_meta == meta and (not force):
        return
    print(f'meta={meta!r} speedup={speedup:.1f} % timing={timing:.3f} ms')
    if speedup < 0:
        return
    device_name = torch.cuda.get_device_name()
    update('scatter_mm', device_name, version, key, tuple((meta[k] for k in sorted(meta))))