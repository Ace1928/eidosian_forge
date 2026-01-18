from threading import Condition
from types import TracebackType
from typing import Dict, List, Optional, Tuple, Type, Union, cast
import torch
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed import rpc
from fairscale.nn.pipe import microbatch
from fairscale.nn.pipe.checkpoint import Checkpointing, TensorOrTensors
from fairscale.nn.pipe.dependency import fork, join
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.stream import as_cuda, current_stream, is_cuda, use_device, use_stream
from fairscale.nn.pipe.worker import Task, create_workers
from .data import DataConsumer
def local_parameter_rrefs(self) -> List[rpc.RRef]:
    """
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
    return [rpc.RRef(p) for p in self.module.parameters()]