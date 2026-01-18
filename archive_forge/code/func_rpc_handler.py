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
def rpc_handler(self, f: Callable):
    """
        Add a function to the server. It will respond to JSON RPC requests with the corresponding method name.
        This can be used as both a side-effecting function or as a decorator.

        :param f: Function to add
        :return: Function wrapper (so it can be used as a decorator)
        """
    return self.rpc_spec.add_handler(f)