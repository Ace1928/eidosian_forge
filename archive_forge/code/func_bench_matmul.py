from typing import Any, Dict
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components.attention.core import SparseCS, _matmul_with_mask
def bench_matmul(dtype: torch.dtype, shapes):
    results: Dict[str, Any] = {}
    Z, H = (1, 1)
    for M, N, K in shapes:
        modes = [(mode, block) for mode in ['sdd', 'dsd'] for block in [16, 32, 64]]
        for mode, block in modes:
            a = torch.randn((Z, H, M, K), dtype=dtype, device='cuda')
            b = torch.randn((Z, H, K, N), dtype=dtype, device='cuda')
            shape = {'sdd': (M, N), 'dsd': (a.shape[2], a.shape[3]), 'dds': (b.shape[2], b.shape[3])}[mode]
            _layout = torch.eye(shape[0] // block, shape[1] // block, dtype=torch.long)
            layout = _layout.unsqueeze(0).expand(H, -1, -1)
            a_triton = triton.testing.sparsify_tensor(a, layout, block) if mode == 'dsd' else a
            b_triton = triton.testing.sparsify_tensor(b, layout, block) if mode == 'dds' else b
            bsmm = blocksparse_matmul(layout=layout, block=block, mode=mode, device=torch.device('cuda'), trans_a=False, trans_b=False)
            ta = triton.testing.mask_tensor(a, layout, block) if mode == 'dsd' else a
            tb = triton.testing.mask_tensor(b, layout, block) if mode == 'dds' else b
            mask = torch.ones_like(a, dtype=torch.float, device='cuda')
            mask = triton.testing.mask_tensor(mask, layout, block, value=0.0)
            a_cs = a.flatten(start_dim=0, end_dim=1).to(torch.float32)
            b_cs = b.flatten(start_dim=0, end_dim=1).to(torch.float32)
            a_cs = a_cs.contiguous()
            b_cs = b_cs.transpose(-2, -1).contiguous()
            if mode == 'sdd':
                b_cs = b_cs.transpose(-2, -1)
            sparse_cs_mask = SparseCS(mask.flatten(start_dim=0, end_dim=1).contiguous(), device=torch.device('cuda'))
            op_flops = {'sdd': 2 * Z * K * float(layout.sum()) * block * block, 'dsd': 2 * Z * N * float(layout.sum()) * block * block, 'dds': 2 * Z * M * float(layout.sum()) * block * block}[mode] * 1e-12

            def torch_step():
                return torch.matmul(ta, tb)

            def triton_step():
                return bsmm(a_triton, b_triton)

            def sparse_step():
                if mode == 'sdd':
                    return _matmul_with_mask(a_cs, b_cs, sparse_cs_mask)
                else:
                    return sparse_cs_mask.spmm(b_cs)
            for testcase in [TestCase(torch_step, f'pytorch - {mode} - {block}: '), TestCase(sparse_step, f'sparse - {mode} - {block}: '), TestCase(triton_step, f'triton  - {mode} - {block}: ')]:
                ms = triton.testing.do_bench(lambda: testcase.function())[0]
                key = f'M={M}, N={N}, K={K}'
                if key not in results:
                    results[key] = {}
                num_flops = op_flops / ms * 1000.0
                results[key][testcase.name] = f'{num_flops:.1f}'
                print(f'{key} - {testcase.name} - {num_flops:.2f}TFlops')
    pretty_print(results, title='\n ------------- Type: {} -------------'.format(dtype), units='TFlops/s')
    pretty_plot(results, title=f'Sparse/Blocksparse throughput - {dtype}', filename=f'blocksparse_{dtype}.png', dash_key='pytorch', units='TFlops/s')