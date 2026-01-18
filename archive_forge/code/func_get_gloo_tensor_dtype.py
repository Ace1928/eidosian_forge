import asyncio
import time
from typing import List
import numpy
import ray
import ray.experimental.internal_kv as internal_kv
from ray._raylet import GcsClient
from ray.util.collective.types import ReduceOp, torch_available
from ray.util.queue import _QueueActor
def get_gloo_tensor_dtype(tensor):
    """Return the corresponded GLOO dtype given a tensor."""
    if isinstance(tensor, numpy.ndarray):
        return NUMPY_GLOO_DTYPE_MAP[tensor.dtype.type]
    if torch_available():
        if isinstance(tensor, torch.Tensor):
            if not tensor.is_cuda:
                return TORCH_GLOO_DTYPE_MAP[tensor.dtype]
            else:
                raise ValueError('Expect torch CPU tensor. Got {}.'.format(tensor.device))
    raise ValueError('Unsupported tensor type. Got: {}.'.format(type(tensor)))