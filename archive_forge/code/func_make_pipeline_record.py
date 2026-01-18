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
def make_pipeline_record(self, consumers: List[DataConsumer]) -> DistributedPipelineRecord:
    return DistributedPipelineRecord(self.device, self.rank, self.chunks, self.num_inputs, self.num_outputs, consumers)