import typing as t
import zmq
from tornado import ioloop
from traitlets import Instance, Type
from zmq.eventloop.zmqstream import ZMQStream
from ..manager import AsyncKernelManager, KernelManager
from .restarter import AsyncIOLoopKernelRestarter, IOLoopKernelRestarter
def as_zmqstream(f: t.Any) -> t.Callable:
    """Convert a socket to a zmq stream."""

    def wrapped(self: t.Any, *args: t.Any, **kwargs: t.Any) -> t.Any:
        save_socket_class = None
        if self.context._socket_class is not zmq.Socket:
            save_socket_class = self.context._socket_class
            self.context._socket_class = zmq.Socket
        try:
            socket = f(self, *args, **kwargs)
        finally:
            if save_socket_class:
                self.context._socket_class = save_socket_class
        return ZMQStream(socket, self.loop)
    return wrapped