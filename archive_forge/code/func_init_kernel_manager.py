import atexit
import os
import signal
import sys
import typing as t
import uuid
import warnings
from jupyter_core.application import base_aliases, base_flags
from traitlets import CBool, CUnicode, Dict, List, Type, Unicode
from traitlets.config.application import boolean_flag
from . import KernelManager, connect, find_connection_file, tunnel_to_kernel
from .blocking import BlockingKernelClient
from .connect import KernelConnectionInfo
from .kernelspec import NoSuchKernel
from .localinterfaces import localhost
from .restarter import KernelRestarter
from .session import Session
from .utils import _filefind
def init_kernel_manager(self) -> None:
    """Initialize the kernel manager."""
    if self.existing:
        self.kernel_manager = None
        return
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        self.kernel_manager = self.kernel_manager_class(ip=self.ip, session=self.session, transport=self.transport, shell_port=self.shell_port, iopub_port=self.iopub_port, stdin_port=self.stdin_port, hb_port=self.hb_port, control_port=self.control_port, connection_file=self.connection_file, kernel_name=self.kernel_name, parent=self, data_dir=self.data_dir)
    except NoSuchKernel:
        self.log.critical('Could not find kernel %s', self.kernel_name)
        self.exit(1)
    self.kernel_manager = t.cast(KernelManager, self.kernel_manager)
    self.kernel_manager.client_factory = self.kernel_client_class
    kwargs = {}
    kwargs['extra_arguments'] = self.kernel_argv
    self.kernel_manager.start_kernel(**kwargs)
    atexit.register(self.kernel_manager.cleanup_ipc_files)
    if self.sshserver:
        self.kernel_manager.write_connection_file()
    km = self.kernel_manager
    self.shell_port = km.shell_port
    self.iopub_port = km.iopub_port
    self.stdin_port = km.stdin_port
    self.hb_port = km.hb_port
    self.control_port = km.control_port
    self.connection_file = km.connection_file
    atexit.register(self.kernel_manager.cleanup_connection_file)