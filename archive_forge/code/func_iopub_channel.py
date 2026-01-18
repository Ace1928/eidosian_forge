import asyncio
from jupyter_client.client import KernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_core.utils import run_sync
from traitlets import Instance, Type, default
from .channels import InProcessChannel, InProcessHBChannel
@property
def iopub_channel(self):
    if self._iopub_channel is None:
        self._iopub_channel = self.iopub_channel_class(self)
    return self._iopub_channel