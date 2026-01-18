import asyncio
from jupyter_client.client import KernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_core.utils import run_sync
from traitlets import Instance, Type, default
from .channels import InProcessChannel, InProcessHBChannel
def kernel_info(self):
    """Request kernel info."""
    msg = self.session.msg('kernel_info_request')
    self._dispatch_to_kernel(msg)
    return msg['header']['msg_id']