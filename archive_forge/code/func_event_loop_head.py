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
def event_loop_head(self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals], event: Optional[Event]) -> None:
    """The event loop for the "head", which first performs the forward pass
        on any applicable layers for this stage, and then enters the common
        `event_loop_inner`"""
    invocations, activations = self.get_invocations_and_activations()
    expected_invocations = len(invocations) * len(batches)
    actual_invocations = 0
    count_per_order = dict()
    for batch in batches:
        inv_count, last_order = self.run_invocations_on_batch(batch, invocations, 0, skip_trackers, activations)
        actual_invocations += inv_count
        count_per_order[last_order] = inv_count
    if actual_invocations < expected_invocations or self.training:
        self.event_loop_inner(expected_invocations, skip_trackers, activations, invocations, count_per_order, already_received=actual_invocations, event=event)