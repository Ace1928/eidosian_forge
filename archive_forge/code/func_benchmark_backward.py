import torch
import torch.utils.benchmark as benchmark
def benchmark_backward(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, '- Backward pass')
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    elif grad.shape != y.shape:
        raise RuntimeError('Grad shape does not match output shape')

    def f(*inputs, y, grad):
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)
    t = benchmark.Timer(stmt='f(*inputs, y=y, grad=grad)', globals={'f': f, 'inputs': inputs, 'y': y, 'grad': grad}, num_threads=torch.get_num_threads())
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return (t, m)