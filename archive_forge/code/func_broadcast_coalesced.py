import warnings
import torch
from torch.cuda import nccl
from torch._utils import _take_tensors, _flatten_dense_tensors, \
from typing import List
def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """Broadcast a sequence of tensors to the specified GPUs.

    Small tensors are first coalesced into a buffer to reduce the number of synchronizations.

    Args:
        tensors (sequence): tensors to broadcast. Must be on the same device,
          either CPU or GPU.
        devices (Iterable[torch.device, str or int]): an iterable of GPU
          devices, among which to broadcast.
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple containing copies of :attr:`tensor`, placed on :attr:`devices`.
    """
    devices = [_get_device_index(d) for d in devices]
    tensors = [_handle_complex(t) for t in tensors]
    return torch._C._broadcast_coalesced(tensors, devices, buffer_size)