import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
def setsockopt_out(self, opt: int, value: Any):
    """Enqueue setsockopt(opt, value) for out_socket

        See zmq.Socket.setsockopt for details.
        """
    self._out_sockopts.append((opt, value))