import os
import sys
import logging
import asyncio
import typing
import contextlib
import threading
import inspect
from itertools import chain
from queue import Empty, Full, Queue, LifoQueue
from urllib.parse import parse_qs, unquote, urlparse, ParseResult
from redis.connection import (
from redis.asyncio.connection import (
import aiokeydb.v2.exceptions as exceptions
from aiokeydb.v2.utils import set_ulimits
def reset_pool(self, inuse_connections: bool=True, raise_exceptions: bool=False):
    """
        Resets the connection pool
        """
    self.disconnect(inuse_connections=inuse_connections, raise_exceptions=raise_exceptions, with_lock=False)
    self.reset()