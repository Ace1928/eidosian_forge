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
def init_sockets(self):
    """Create a context, a session, and the kernel sockets."""
    self.log.info('Starting the kernel at pid: %i', os.getpid())
    assert self.context is None, 'init_sockets cannot be called twice!'
    self.context = context = zmq.Context()
    atexit.register(self.close)
    self.shell_socket = context.socket(zmq.ROUTER)
    self.shell_socket.linger = 1000
    self.shell_port = self._bind_socket(self.shell_socket, self.shell_port)
    self.log.debug('shell ROUTER Channel on port: %i' % self.shell_port)
    self.stdin_socket = context.socket(zmq.ROUTER)
    self.stdin_socket.linger = 1000
    self.stdin_port = self._bind_socket(self.stdin_socket, self.stdin_port)
    self.log.debug('stdin ROUTER Channel on port: %i' % self.stdin_port)
    if hasattr(zmq, 'ROUTER_HANDOVER'):
        self.shell_socket.router_handover = self.stdin_socket.router_handover = 1
    self.init_control(context)
    self.init_iopub(context)