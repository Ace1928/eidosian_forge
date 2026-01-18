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
def is_stale(self, cached):
    ts = time.time()
    if cached is None:
        return True
    if ts - cached.ts > self._grace:
        return True
    if cached.pol_body['max_age'] + cached.ts < ts:
        return True
    return False