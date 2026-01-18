import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
def bind_out_to_random_port(self, addr: str, *args, **kwargs) -> int:
    """Enqueue a random port on the given interface for binding on
        out_socket.

        See zmq.Socket.bind_to_random_port for details.

        .. versionadded:: 18.0
        """
    port = self._reserve_random_port(addr, *args, **kwargs)
    self.bind_out('%s:%i' % (addr, port))
    return port