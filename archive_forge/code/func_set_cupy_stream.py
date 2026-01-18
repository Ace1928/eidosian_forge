import contextlib
import torch
from torch.distributed import ReduceOp
@contextlib.contextmanager
def set_cupy_stream(stream: torch.cuda.Stream):
    """Set the cuda stream for communication"""
    cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream, stream.device_index)
    with cupy_stream:
        yield