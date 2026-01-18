from typing import Any, Callable, List, no_type_check
import torch
import torch.distributed as dist
from torch.autograd import Variable
from functools import partial
from dataclasses import dataclass
def apply_optim_in_backward_hook(hook_state: Any, bucket: dist.GradBucket, optim_stream_state) -> torch.futures.Future[torch.Tensor]:
    ddp_weakref = hook_state
    ddp_inst = ddp_weakref()
    reducer, process_group = (ddp_inst.reducer, ddp_inst.process_group)
    fut = reducer._run_allreduce_hook(bucket)
    optimizer_stream = optim_stream_state.optim_stream
    with torch.cuda.stream(optimizer_stream):
        fut.wait()
        bucket.buffer().div_(process_group.size())
        model_params = bucket.parameters()
        grads = bucket.gradients()
        for p, g in zip(model_params, grads):
            if hasattr(p, '_in_backward_optimizers'):
                if not gradient_is_bucket_view:
                    p.grad = g
                for optim in p._in_backward_optimizers:
                    optim.step()
    ret_fut = torch.futures.Future()
    ret_fut.set_result(bucket.buffer())

    def wait_for_optim_stream_callback():
        torch.cuda.current_stream().wait_stream(optim_stream_state.optim_stream)
        for param in ddp_inst._get_data_parallel_params(ddp_inst.module):
            if hasattr(param, '_in_backward_optimizers'):
                param.grad = None
        optim_stream_state.wait_for_optim_stream_enqueued = False
    if not optim_stream_state.wait_for_optim_stream_enqueued:
        Variable._execution_engine.queue_callback(wait_for_optim_stream_callback)
        optim_stream_state.wait_for_optim_stream_enqueued = True
    return ret_fut