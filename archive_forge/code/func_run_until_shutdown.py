from __future__ import annotations
import logging # isort:skip
import atexit
import signal
import socket
import sys
from types import FrameType
from typing import TYPE_CHECKING, Any, Mapping
from tornado import netutil, version as tornado_version
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from .. import __version__
from ..core import properties as p
from ..core.properties import (
from ..resources import DEFAULT_SERVER_PORT, server_url
from ..util.options import Options
from .tornado import DEFAULT_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES, BokehTornado
from .util import bind_sockets, create_hosts_allowlist
def run_until_shutdown(self) -> None:
    """ Run the Bokeh Server until shutdown is requested by the user,
        either via a Keyboard interrupt (Ctrl-C) or SIGTERM.

        Calling this method will start the Tornado ``IOLoop`` and block
        all execution in the calling process.

        Returns:
            None

        """
    if not self._started:
        self.start()
    atexit.register(self._atexit)
    signal.signal(signal.SIGTERM, self._sigterm)
    try:
        self._loop.start()
    except KeyboardInterrupt:
        print('\nInterrupted, shutting down')
    self.stop()