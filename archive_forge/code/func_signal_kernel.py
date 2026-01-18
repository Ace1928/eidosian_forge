from jupyter_client.manager import KernelManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_client.session import Session
from traitlets import DottedObjectName, Instance, default
from .constants import INPROCESS_KEY
def signal_kernel(self, signum):
    """Send a signal to the kernel."""
    msg = 'Cannot signal in-process kernel.'
    raise NotImplementedError(msg)