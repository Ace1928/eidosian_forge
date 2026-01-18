import time
from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn
from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from fairscale.nn.model_parallel import get_pipeline_parallel_ranks
from fairscale.nn.pipe.async_schedule import (
from fairscale.nn.pipe.checkpoint import Checkpointing
from fairscale.nn.pipe.messages import Transport
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.types import (
from fairscale.nn.pipe.worker import Task
def async_grad_inner(self, message: PipeMessage, activations: Dict[int, Batch]) -> None:
    args: AsyncMessageBody = message.args
    recvd_grads = self.transport.recv_message_tensors(message)
    batch = activations[args.microbatch_index]
    if len(recvd_grads.tensors) != len(batch):
        raise RuntimeError('different number of tensors and gradients')
    grads = []
    final_tensors = []
    for i, tensor in enumerate(batch):
        if tensor.requires_grad or getattr(tensor, 'grad_fn', None) is not None:
            grads.append(recvd_grads.tensors[i])
            final_tensors.append(tensor)
    torch.autograd.backward(final_tensors, grad_tensors=grads, retain_graph=True)
    del activations[args.microbatch_index]