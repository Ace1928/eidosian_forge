import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
def connect_out(self, addr: str):
    """Enqueue ZMQ address for connecting on out_socket.

        See zmq.Socket.connect for details.
        """
    self._out_connects.append(addr)