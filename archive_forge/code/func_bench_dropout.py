from typing import Any, Dict, Optional
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation, build_activation
from xformers.triton import FusedDropoutBias
def bench_dropout(bias: bool, backward: bool, activation: Optional[Activation]):
    device = torch.device('cuda')
    for dtype in [torch.float16, torch.float32]:
        results: Dict[str, Any] = {}
        for B, M, K in SHAPES:
            a = torch.rand((B, M, K), device=device, dtype=dtype, requires_grad=backward)
            b = torch.rand(K, device=device, dtype=dtype, requires_grad=backward)
            torch_act = build_activation(activation)
            triton_dropout = FusedDropoutBias(P, bias_shape=K if bias else None, activation=activation)

            def torch_step(x):
                x_ = x + b if bias else x
                y = torch.nn.functional.dropout(x_, P)
                if activation:
                    y = torch_act(y)
                if backward:
                    y.grad = None
                    torch.norm(y).backward()
                return y

            def triton_step(x):
                y = triton_dropout(x)
                if backward:
                    y.grad = None
                    torch.norm(y).backward()
                return y
            for testcase in [TestCase(torch_step, 'pytorch - bias: {} - fw{} - act: {}'.format(bias, '+bw' if backward else '', activation)), TestCase(triton_step, 'triton - bias: {} - fw{} - act: {}'.format(bias, '+bw' if backward else '', activation))]:
                time = triton.testing.do_bench(lambda: testcase.function(a), grad_to_none=[a, b])[0]
                key = f'B={B}, M={M}, K={K}'
                if key not in results:
                    results[key] = {}
                bandwidth = to_gbs_fw(a, time, bias)
                results[key][testcase.name] = f'{bandwidth:.1f}'
        pretty_print(results, title='\n --- Type: {} --- '.format(dtype), units='GB/s')
        pretty_plot(results, title='Dropout-Bias-{}-FW{}-{}-Act: {}'.format(bias, '+BW' if backward else '', dtype, activation), units='GB/s', dash_key='pytorch')