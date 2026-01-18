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
def run_invocation(self, batch: Batch, partition: ModuleWrapper, skip_trackers: List[SkipTrackerThroughPotals], invocation: Invocation) -> Batch:
    """Actually run the forward pass for a given module, and send the result
        to the next stage in the pipeline if needed."""
    task = create_task(self.checkpoint_stop, batch.index, self.group.rank(), batch, partition.module, skip_trackers)
    result = task.compute()
    task.finalize(result)
    if invocation.dest and invocation.dest.stage != invocation.this.stage:
        ranks = get_pipeline_parallel_ranks()
        dst_rank = ranks[invocation.dest.stage]
        result = self.send_async_message(dst_rank, result, invocation)
    return result