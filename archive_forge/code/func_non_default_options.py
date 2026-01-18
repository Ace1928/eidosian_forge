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
@property
def non_default_options(self) -> dict[str, Any]:
    """The non-default options this pool was created with.

        Added for CMAP's :class:`PoolCreatedEvent`.
        """
    opts = {}
    if self.__max_pool_size != MAX_POOL_SIZE:
        opts['maxPoolSize'] = self.__max_pool_size
    if self.__min_pool_size != MIN_POOL_SIZE:
        opts['minPoolSize'] = self.__min_pool_size
    if self.__max_idle_time_seconds != MAX_IDLE_TIME_SEC:
        assert self.__max_idle_time_seconds is not None
        opts['maxIdleTimeMS'] = self.__max_idle_time_seconds * 1000
    if self.__wait_queue_timeout != WAIT_QUEUE_TIMEOUT:
        assert self.__wait_queue_timeout is not None
        opts['waitQueueTimeoutMS'] = self.__wait_queue_timeout * 1000
    if self.__max_connecting != MAX_CONNECTING:
        opts['maxConnecting'] = self.__max_connecting
    return opts