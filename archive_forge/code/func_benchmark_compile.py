import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils.benchmark import Timer
from typing import Optional, List, Callable, Union, Any, cast
def benchmark_compile(model: Union[torch.nn.Module, Callable], sample_input: Union[torch.Tensor, Any], num_iters: int=5, backend: Optional[str]=None, mode: Optional[str]='default', optimizer: Optional[torch.optim.Optimizer]=None, loss_fn: Union[torch.nn.Module, Callable, None]=None):
    """
        Use this utility to benchmark torch.compile
        """
    if backend:
        try:
            torch._dynamo.reset()
            compile_counter_with_backend = CompileCounterWithBackend(backend)
            opt_model = torch.compile(model, backend=compile_counter_with_backend, mode=mode)
            compilation_time = bench_loop(opt_model, sample_input, 1, optimizer, loss_fn)
            running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)
            if compile_counter_with_backend.frame_count == 0:
                raise RuntimeError('No compilation occurred during benchmarking.')
            if compile_counter_with_backend.frame_count > 1:
                raise RuntimeError('Recompilation occurred during benchmarking.')
        except Exception as e:
            print(e)
            print(f'Failed to compile {backend} with mode {mode}')
            return (None, None)
    else:
        opt_model = model
        compilation_time = None
        running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)
    compilation_time = round(compilation_time, 2) if compilation_time else None
    running_time = round(running_time, 2) if running_time else None
    return (compilation_time, running_time)