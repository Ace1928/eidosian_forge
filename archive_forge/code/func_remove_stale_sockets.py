from __future__ import annotations
import collections
import contextlib
import copy
import os
import platform
import socket
import ssl
import sys
import threading
import time
import weakref
from typing import (
import bson
from bson import DEFAULT_CODEC_OPTIONS
from bson.son import SON
from pymongo import __version__, _csot, auth, helpers
from pymongo.client_session import _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.lock import _create_lock
from pymongo.monitoring import (
from pymongo.network import command, receive_message
from pymongo.read_preferences import ReadPreference
from pymongo.server_api import _add_to_command
from pymongo.server_type import SERVER_TYPE
from pymongo.socket_checker import SocketChecker
from pymongo.ssl_support import HAS_SNI, SSLError
def remove_stale_sockets(self, reference_generation: int) -> None:
    """Removes stale sockets then adds new ones if pool is too small and
        has not been reset. The `reference_generation` argument specifies the
        `generation` at the point in time this operation was requested on the
        pool.
        """
    with self.lock:
        if self.state != PoolState.READY:
            return
    if self.opts.max_idle_time_seconds is not None:
        with self.lock:
            while self.conns and self.conns[-1].idle_time_seconds() > self.opts.max_idle_time_seconds:
                conn = self.conns.pop()
                conn.close_conn(ConnectionClosedReason.IDLE)
    while True:
        with self.size_cond:
            if len(self.conns) + self.active_sockets >= self.opts.min_pool_size:
                return
            if self.requests >= self.opts.min_pool_size:
                return
            self.requests += 1
        incremented = False
        try:
            with self._max_connecting_cond:
                if self._pending >= self._max_connecting:
                    return
                self._pending += 1
                incremented = True
            conn = self.connect()
            with self.lock:
                if self.gen.get_overall() != reference_generation:
                    conn.close_conn(ConnectionClosedReason.STALE)
                    return
                self.conns.appendleft(conn)
        finally:
            if incremented:
                with self._max_connecting_cond:
                    self._pending -= 1
                    self._max_connecting_cond.notify()
            with self.size_cond:
                self.requests -= 1
                self.size_cond.notify()