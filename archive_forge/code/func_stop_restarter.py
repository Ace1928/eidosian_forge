import typing as t
import zmq
from tornado import ioloop
from traitlets import Instance, Type
from zmq.eventloop.zmqstream import ZMQStream
from ..manager import AsyncKernelManager, KernelManager
from .restarter import AsyncIOLoopKernelRestarter, IOLoopKernelRestarter
def stop_restarter(self) -> None:
    """Stop the restarter."""
    if self.autorestart and self._restarter is not None:
        self._restarter.stop()