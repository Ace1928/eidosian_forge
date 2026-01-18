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
def event_loop_trunk_backward_helper(self, activations: Dict[int, Batch]) -> None:
    message = self.transport.recv_message_header(EVENT_LOOP_GRADIENTS_QUEUE)
    args: AsyncMessageBody = message.args
    assert args.message_type is AsyncMessageType.Gradients
    self.async_grad_inner(message, activations)