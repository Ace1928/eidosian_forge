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
def event_loop_tail(self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals]) -> None:
    """The event loop for the "tail", or final stage which only processes
        activations and then returns to the caller so that the loss can be
        calculated. This also handles the first/only stage for the special
        case of a 1-stage pipeline."""
    invocations, activations = self.get_invocations_and_activations()
    expected_invocations = len(invocations) * len(batches)
    actual_invocations = 0
    rank = self.group.rank()
    count_per_order = dict()
    for batch in batches:
        if rank == 0:
            order = 0
        else:
            message = self.transport.recv_message_header(EVENT_LOOP_QUEUE)
            args: AsyncMessageBody = message.args
            batch = self.get_batch_from_message(message)
            order = args.order
        inv_count, last_order = self.run_invocations_on_batch(batch, invocations, order, skip_trackers, activations)
        actual_invocations += inv_count
        count_per_order[last_order] = inv_count
        if invocations[last_order].dest is None:
            self.prepare_tail_backward(batch, activations, invocations, count_per_order, len(invocations) - inv_count)
    if actual_invocations < expected_invocations:
        expected_gradients = 0
        self.event_loop_inner(expected_invocations, skip_trackers, activations, invocations, count_per_order, already_received=actual_invocations, ignore_gradients=True, tail=True)
    _, last_invocation = invocations.popitem()
    for index, batch in activations[len(self.partitions) - 1][last_invocation.order].items():
        batches[index] = batch