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
def reqrep(wrapped: t.Callable, meth: t.Callable, channel: str='shell') -> t.Callable:
    wrapped = wrapped(meth, channel)
    if not meth.__doc__:
        return wrapped
    basedoc, _ = meth.__doc__.split('Returns\n', 1)
    parts = [basedoc.strip()]
    if 'Parameters' not in basedoc:
        parts.append('\n        Parameters\n        ----------\n        ')
    parts.append('\n        reply: bool (default: False)\n            Whether to wait for and return reply\n        timeout: float or None (default: None)\n            Timeout to use when waiting for a reply\n\n        Returns\n        -------\n        msg_id: str\n            The msg_id of the request sent, if reply=False (default)\n        reply: dict\n            The reply message for this request, if reply=True\n    ')
    wrapped.__doc__ = '\n'.join(parts)
    return wrapped