import asyncio
from jupyter_client.client import KernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_core.utils import run_sync
from traitlets import Instance, Type, default
from .channels import InProcessChannel, InProcessHBChannel
def comm_info(self, target_name=None):
    """Request a dictionary of valid comms and their targets."""
    content = {} if target_name is None else dict(target_name=target_name)
    msg = self.session.msg('comm_info_request', content)
    self._dispatch_to_kernel(msg)
    return msg['header']['msg_id']