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
def init_control(self, context):
    """Initialize the control channel."""
    self.control_socket = context.socket(zmq.ROUTER)
    self.control_socket.linger = 1000
    self.control_port = self._bind_socket(self.control_socket, self.control_port)
    self.log.debug('control ROUTER Channel on port: %i' % self.control_port)
    self.debugpy_socket = context.socket(zmq.STREAM)
    self.debugpy_socket.linger = 1000
    self.debug_shell_socket = context.socket(zmq.DEALER)
    self.debug_shell_socket.linger = 1000
    if self.shell_socket.getsockopt(zmq.LAST_ENDPOINT):
        self.debug_shell_socket.connect(self.shell_socket.getsockopt(zmq.LAST_ENDPOINT))
    if hasattr(zmq, 'ROUTER_HANDOVER'):
        self.control_socket.router_handover = 1
    self.control_thread = ControlThread(daemon=True)