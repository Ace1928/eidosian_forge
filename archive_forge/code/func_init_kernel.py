from __future__ import annotations
import atexit
import errno
import logging
import os
import signal
import sys
import traceback
import typing as t
from functools import partial
from io import FileIO, TextIOWrapper
from logging import StreamHandler
from pathlib import Path
import zmq
from IPython.core.application import (  # type:ignore[attr-defined]
from IPython.core.profiledir import ProfileDir
from IPython.core.shellapp import InteractiveShellApp, shell_aliases, shell_flags
from jupyter_client.connect import ConnectionFileMixin
from jupyter_client.session import Session, session_aliases, session_flags
from jupyter_core.paths import jupyter_runtime_dir
from tornado import ioloop
from traitlets.traitlets import (
from traitlets.utils import filefind
from traitlets.utils.importstring import import_item
from zmq.eventloop.zmqstream import ZMQStream
from .connect import get_connection_info, write_connection_file
from .control import ControlThread
from .heartbeat import Heartbeat
from .iostream import IOPubThread
from .ipkernel import IPythonKernel
from .parentpoller import ParentPollerUnix, ParentPollerWindows
from .zmqshell import ZMQInteractiveShell
def init_kernel(self):
    """Create the Kernel object itself"""
    shell_stream = ZMQStream(self.shell_socket)
    control_stream = ZMQStream(self.control_socket, self.control_thread.io_loop)
    debugpy_stream = ZMQStream(self.debugpy_socket, self.control_thread.io_loop)
    self.control_thread.start()
    kernel_factory = self.kernel_class.instance
    kernel = kernel_factory(parent=self, session=self.session, control_stream=control_stream, debugpy_stream=debugpy_stream, debug_shell_socket=self.debug_shell_socket, shell_stream=shell_stream, control_thread=self.control_thread, iopub_thread=self.iopub_thread, iopub_socket=self.iopub_socket, stdin_socket=self.stdin_socket, log=self.log, profile_dir=self.profile_dir, user_ns=self.user_ns)
    kernel.record_ports({name + '_port': port for name, port in self._ports.items()})
    self.kernel = kernel
    self.displayhook.get_execution_count = lambda: kernel.execution_count