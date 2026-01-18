from jupyter_client.manager import KernelManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_client.session import Session
from traitlets import DottedObjectName, Instance, default
from .constants import INPROCESS_KEY
def restart_kernel(self, now=False, **kwds):
    """Restart the kernel."""
    self.shutdown_kernel()
    self.start_kernel(**kwds)