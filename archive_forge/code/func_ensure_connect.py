import atexit
import ssl
from pyVim.connect import Disconnect, SmartStubAdapter, VimSessionOrientedStub
from pyVmomi import vim
from ray.autoscaler._private.vsphere.utils import Constants, singleton_client
def ensure_connect(self):
    try:
        _ = self.pyvmomi_sdk_client.RetrieveContent()
    except vim.fault.NotAuthenticated:
        self.pyvmomi_sdk_client = self.get_client()
    except Exception as e:
        raise RuntimeError(f'failed to ensure the connect, exception: {e}')