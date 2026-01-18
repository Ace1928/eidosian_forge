import os
import signal
import typing as t
import uuid
from jupyter_core.application import JupyterApp, base_flags
from tornado.ioloop import IOLoop
from traitlets import Unicode
from . import __version__
from .kernelspec import NATIVE_KERNEL_NAME, KernelSpecManager
from .manager import KernelManager
def log_connection_info(self) -> None:
    """Log the connection info for the kernel."""
    cf = self.km.connection_file
    self.log.info('Connection file: %s', cf)
    self.log.info('To connect a client: --existing %s', os.path.basename(cf))