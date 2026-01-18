import asyncio
import inspect
import sys
import time
import typing as t
from functools import partial
from getpass import getpass
from queue import Empty
import zmq.asyncio
from jupyter_core.utils import ensure_async
from traitlets import Any, Bool, Instance, Type
from .channels import major_protocol_version
from .channelsabc import ChannelABC, HBChannelABC
from .clientabc import KernelClientABC
from .connect import ConnectionFileMixin
from .session import Session
@property
def channels_running(self) -> bool:
    """Are any of the channels created and running?"""
    return self._shell_channel and self.shell_channel.is_alive() or (self._iopub_channel and self.iopub_channel.is_alive()) or (self._stdin_channel and self.stdin_channel.is_alive()) or (self._hb_channel and self.hb_channel.is_alive()) or (self._control_channel and self.control_channel.is_alive())