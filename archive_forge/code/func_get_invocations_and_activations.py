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
def get_invocations_and_activations(self) -> Tuple[Invocations, Activations]:
    activations: Activations = dict()
    invocations: Invocations = OrderedDict()
    for pi, partition in enumerate(self.partitions):
        activations[pi] = dict()
        for invocation in partition.invocations:
            activations[pi][invocation.order] = dict()
            invocations[invocation.order] = invocation
    invocations = OrderedDict(sorted(invocations.items(), key=lambda entry: entry[0]))
    return (invocations, activations)