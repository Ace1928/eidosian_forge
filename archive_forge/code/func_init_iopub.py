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
def init_iopub(self, context):
    """Initialize the iopub channel."""
    self.iopub_socket = context.socket(zmq.PUB)
    self.iopub_socket.linger = 1000
    self.iopub_port = self._bind_socket(self.iopub_socket, self.iopub_port)
    self.log.debug('iopub PUB Channel on port: %i' % self.iopub_port)
    self.configure_tornado_logger()
    self.iopub_thread = IOPubThread(self.iopub_socket, pipe=True)
    self.iopub_thread.start()
    self.iopub_socket = self.iopub_thread.background_socket