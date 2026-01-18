import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
def connect_in(self, addr: str) -> None:
    """Enqueue ZMQ address for connecting on in_socket.

        See zmq.Socket.connect for details.
        """
    self._in_connects.append(addr)