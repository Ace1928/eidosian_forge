import asyncio
import logging
import sys
import time
from typing import Dict, Optional, Union
from warnings import warn
import zmq
import zmq.asyncio
from rpcq._base import to_msgpack, from_msgpack
import rpcq._utils as utils
from rpcq.messages import RPCError, RPCReply
@property
def auth_configured(self) -> bool:
    return self._auth_config is not None