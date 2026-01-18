import typing as t
import zmq
from tornado import ioloop
from traitlets import Instance, Type
from zmq.eventloop.zmqstream import ZMQStream
from ..manager import AsyncKernelManager, KernelManager
from .restarter import AsyncIOLoopKernelRestarter, IOLoopKernelRestarter
def start_restarter(self) -> None:
    """Start the restarter."""
    if self.autorestart and self.has_kernel:
        if self._restarter is None:
            self._restarter = self.restarter_class(kernel_manager=self, loop=self.loop, parent=self, log=self.log)
        self._restarter.start()