import asyncio
import logging
import time
import collections
import sys
import os
import socket
from functools import partial
from .resolver import STSResolver, STSFetchResult
from .constants import QUEUE_LIMIT, CHUNK, REQUEST_LIMIT
from .utils import create_custom_socket, filter_domain, is_ipaddr
from .base_cache import CacheEntry
from . import netstring
def cleanup_queue():
    while not queue.empty():
        task = queue.get_nowait()
        try:
            task.cancel()
        except Exception:
            pass