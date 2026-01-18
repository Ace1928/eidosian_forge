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
def async_send_inner(self, batch: Batch, index: int) -> Tuple[Batch, PipeMessage]:
    task = create_task_without_skip_trackers(self.checkpoint_stop, index, self.group.rank(), batch, self.partitions[0].module)
    result = task.compute()
    task.finalize(result)
    ranks = get_pipeline_parallel_ranks()
    this_rank = torch.distributed.get_rank()
    body = AsyncMessageBody(AsyncMessageType.Activations, index, Location(this_rank, 0), Location(ranks[ranks.index(this_rank) + 1], 0), 0)
    message = PipeMessage(this_rank, ranks[ranks.index(this_rank) + 1], queue_name=EVENT_LOOP_ACTIVATIONS_QUEUE, args=body, tensors=tuple([*result]))
    return (result, message)