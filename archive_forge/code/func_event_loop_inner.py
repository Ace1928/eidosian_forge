from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from threading import Event
from typing import Dict, Iterable, List, Optional, Tuple
import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup
from fairscale.nn.model_parallel import get_pipeline_parallel_ranks
from .checkpoint import Checkpointing
from .messages import Transport
from .microbatch import Batch
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors, Tensors
from .worker import Task
def event_loop_inner(self, expected_invocations: int, skip_trackers: List[SkipTrackerThroughPotals], activations: Activations, invocations: Invocations, count_per_order: Dict[int, int], *, already_received: int=0, ignore_gradients: bool=False, event: Optional[Event]=None, tail: bool=False) -> None:
    """The common event loop shared by all stages. This processses
        activations for the forward pass, and if `self.training` is true,
        processes gradients for the backward pass."""
    num_activations = already_received
    if self.training and (not ignore_gradients):
        num_gradients = 0
    else:
        num_gradients = expected_invocations
    while num_activations < expected_invocations or num_gradients < expected_invocations:
        if num_activations == expected_invocations and num_gradients == 0 and (event is not None):
            event.wait()
        message = self.transport.recv_message_header(EVENT_LOOP_QUEUE)
        args: AsyncMessageBody = message.args
        invocation = invocations[args.order]
        if args.message_type is AsyncMessageType.Activations:
            batch = self.get_batch_from_message(message)
            inv_count, last_order = self.run_invocations_on_batch(batch, invocations, args.order, skip_trackers, activations)
            count_per_order[last_order] = inv_count
            num_activations += inv_count
            if tail and invocations[last_order].dest is None:
                self.prepare_tail_backward(batch, activations, invocations, count_per_order, len(invocations) - inv_count)
            assert num_activations <= expected_invocations
        elif args.message_type is AsyncMessageType.Gradients:
            num_gradients += count_per_order[invocation.order]
            self.perform_backward_for_invocation(self.transport, message, activations, invocation)