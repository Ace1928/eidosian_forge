import ctypes
import torch
from torch._streambase import _EventBase, _StreamBase
from ._utils import _dummy_type
@classmethod
def from_ipc_handle(cls, device, handle):
    """Reconstruct an event from an IPC handle on the given device."""
    return super().from_ipc_handle(device, handle)