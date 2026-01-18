from jupyter_client.manager import KernelManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_client.session import Session
from traitlets import DottedObjectName, Instance, default
from .constants import INPROCESS_KEY
def shutdown_kernel(self):
    """Shutdown the kernel."""
    if self.kernel:
        self.kernel.iopub_thread.stop()
        self._kill_kernel()