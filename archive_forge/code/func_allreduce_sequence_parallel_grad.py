from typing import Optional
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
def allreduce_sequence_parallel_grad(model: torch.nn.Module, process_group: ProcessGroup):
    params_seqparallel = {name: p for name, p in model.named_parameters() if getattr(p, '_sequence_parallel', False)}
    grads = [p.grad for _, p in sorted(params_seqparallel.items())]
    if grads:
        with torch.no_grad():
            coalesced = torch._utils._flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=process_group)
            for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)