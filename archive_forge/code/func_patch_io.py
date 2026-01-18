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
def patch_io(self):
    """Patch important libraries that can't handle sys.stdout forwarding"""
    try:
        import faulthandler
    except ImportError:
        pass
    else:
        faulthandler_enable = faulthandler.enable

        def enable(file=sys.__stderr__, all_threads=True, **kwargs):
            return faulthandler_enable(file=file, all_threads=all_threads, **kwargs)
        faulthandler.enable = enable
        if hasattr(faulthandler, 'register'):
            faulthandler_register = faulthandler.register

            def register(signum, file=sys.__stderr__, all_threads=True, chain=False, **kwargs):
                return faulthandler_register(signum, file=file, all_threads=all_threads, chain=chain, **kwargs)
            faulthandler.register = register