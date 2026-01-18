import asyncio
import logging
import sys
from asyncio import AbstractEventLoop
from typing import Callable, List, Optional, Tuple
from datetime import datetime
import zmq.asyncio
from zmq.auth.asyncio import AsyncioAuthenticator
from rpcq._base import to_msgpack, from_msgpack
from rpcq._spec import RPCSpec
from rpcq.messages import RPCRequest
def stop_auth(self) -> bool:
    """
        Stops the ZMQ auth service thread, allowing NULL authenticated clients (only) to connect to
            all threads within its context
        """
    if self._authenticator:
        self._socket.curve_server = False
        self._authenticator.stop()
        return True
    else:
        return False