from __future__ import annotations
import os
import queue
import random
import time
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, cast
from pymongo import _csot, common, helpers, periodic_executor
from pymongo.client_session import _ServerSession, _ServerSessionPool
from pymongo.errors import (
from pymongo.hello import Hello
from pymongo.lock import _create_lock
from pymongo.monitor import SrvMonitor
from pymongo.pool import Pool, PoolOptions
from pymongo.server import Server
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import (
from pymongo.topology_description import (
def select_servers(self, selector: Callable[[Selection], Selection], server_selection_timeout: Optional[float]=None, address: Optional[_Address]=None) -> list[Server]:
    """Return a list of Servers matching selector, or time out.

        :Parameters:
          - `selector`: function that takes a list of Servers and returns
            a subset of them.
          - `server_selection_timeout` (optional): maximum seconds to wait.
            If not provided, the default value common.SERVER_SELECTION_TIMEOUT
            is used.
          - `address`: optional server address to select.

        Calls self.open() if needed.

        Raises exc:`ServerSelectionTimeoutError` after
        `server_selection_timeout` if no matching servers are found.
        """
    if server_selection_timeout is None:
        server_timeout = self.get_server_selection_timeout()
    else:
        server_timeout = server_selection_timeout
    with self._lock:
        server_descriptions = self._select_servers_loop(selector, server_timeout, address)
        return [cast(Server, self.get_server_by_address(sd.address)) for sd in server_descriptions]