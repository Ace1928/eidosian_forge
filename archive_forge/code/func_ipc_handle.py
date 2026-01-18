import ctypes
import torch
from torch._streambase import _EventBase, _StreamBase
from ._utils import _dummy_type
def ipc_handle(self):
    """Return an IPC handle of this event.

        If not recorded yet, the event will use the current device.
        """
    return super().ipc_handle()