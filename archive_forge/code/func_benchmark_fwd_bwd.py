import torch
import torch.utils.benchmark as benchmark
def benchmark_fwd_bwd(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    return (benchmark_forward(fn, *inputs, repeats=repeats, desc=desc, verbose=verbose, amp=amp, amp_dtype=amp_dtype, **kwinputs), benchmark_backward(fn, *inputs, grad=grad, repeats=repeats, desc=desc, verbose=verbose, amp=amp, amp_dtype=amp_dtype, **kwinputs))