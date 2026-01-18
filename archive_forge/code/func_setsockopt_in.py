import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
def setsockopt_in(self, opt: int, value: Any) -> None:
    """Enqueue setsockopt(opt, value) for in_socket

        See zmq.Socket.setsockopt for details.
        """
    self._in_sockopts.append((opt, value))