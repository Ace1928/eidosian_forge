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
def init_connection_file(self) -> None:
    """find the connection file, and load the info if found.

        The current working directory and the current profile's security
        directory will be searched for the file if it is not given by
        absolute path.

        When attempting to connect to an existing kernel and the `--existing`
        argument does not match an existing file, it will be interpreted as a
        fileglob, and the matching file in the current profile's security dir
        with the latest access time will be used.

        After this method is called, self.connection_file contains the *full path*
        to the connection file, never just its name.
        """
    runtime_dir = self.runtime_dir
    if self.existing:
        try:
            cf = find_connection_file(self.existing, ['.', runtime_dir])
        except Exception:
            self.log.critical('Could not find existing kernel connection file %s', self.existing)
            self.exit(1)
        self.log.debug('Connecting to existing kernel: %s', cf)
        self.connection_file = cf
    else:
        try:
            cf = find_connection_file(self.connection_file, [runtime_dir])
        except Exception:
            if self.connection_file == os.path.basename(self.connection_file):
                cf = os.path.join(runtime_dir, self.connection_file)
            else:
                cf = self.connection_file
            self.connection_file = cf
    try:
        self.connection_file = _filefind(self.connection_file, ['.', runtime_dir])
    except OSError:
        self.log.debug('Connection File not found: %s', self.connection_file)
        return
    try:
        self.load_connection_file()
    except Exception:
        self.log.error('Failed to load connection file: %r', self.connection_file, exc_info=True)
        self.exit(1)