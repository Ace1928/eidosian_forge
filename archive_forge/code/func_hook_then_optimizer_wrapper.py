from typing import Any, Callable, List, no_type_check
import torch
import torch.distributed as dist
from torch.autograd import Variable
from functools import partial
from dataclasses import dataclass
def hook_then_optimizer_wrapper(hook_state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    fut = hook(hook_state, bucket)

    def optimizer_step(fut):
        gradient_tensors = bucket.gradients()
        model_params = bucket.parameters()
        for grad_tensor, model_param in zip(gradient_tensors, model_params):
            if not has_set_params or model_param in optimizer_state.params_to_optimize:
                optimizer_state.functional_optimizer.step_param(model_param, grad_tensor)
        return bucket.buffer()
    return fut.then(optimizer_step)