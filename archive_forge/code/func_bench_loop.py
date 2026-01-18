import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils.benchmark import Timer
from typing import Optional, List, Callable, Union, Any, cast
def bench_loop(model: Union[torch.nn.Module, Callable], sample_input: Union[torch.Tensor, Any], num_iters: int=5, optimizer: Optional[torch.optim.Optimizer]=None, loss_fn: Optional[Callable]=None):
    if optimizer and loss_fn:
        stmt = '\n    output = model(sample_input)\n    loss = loss_fn(output) if loss_fn else output.sum()\n    loss.backward()\n    optimizer.step()\n    optimizer.zero_grad()\n            '
    else:
        stmt = 'model(sample_input)'
    timer = Timer(stmt=stmt, globals={'model': model, 'sample_input': sample_input, 'optimizer': optimizer, 'loss_fn': loss_fn})
    result = timer.timeit(number=num_iters)
    avg_time = result.mean * 1000
    return round(avg_time, 2)