from jupyter_client.manager import KernelManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_client.session import Session
from traitlets import DottedObjectName, Instance, default
from .constants import INPROCESS_KEY
@property
def has_kernel(self):
    return self.kernel is not None