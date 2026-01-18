from abc import ABC
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Dict, List, Optional
import torch
from fairscale.internal.object import pyobject_to_tensor, tensor_to_pyobject
from fairscale.nn.model_parallel import get_pipeline_parallel_group
from .types import MESSAGE_GENERATION_START, InputDevice, PipeMessage, Tensors
def recv_message_tensors(self, message: PipeMessage) -> PipeMessage:
    torch.cuda.current_stream().synchronize()
    message_tensors = []
    for index, (shape, dtype) in enumerate(zip(message.tensor_shapes, message.tensor_dtypes)):
        t = torch.empty(*shape, dtype=dtype, device=self.input_device)
        torch.distributed.recv(t, message.src, tag=message.tag + index, group=get_pipeline_parallel_group())
        message_tensors.append(t)
    message.tensors = tuple(message_tensors)
    torch.cuda.current_stream().synchronize()
    return message