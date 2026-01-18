import re
import warnings
from typing import Optional, no_type_check
from urllib.parse import urlparse
from tornado import ioloop, web
from tornado.iostream import IOStream
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.utils import JupyterServerAuthWarning
@property
def ping_interval(self):
    """The interval for websocket keep-alive pings.

        Set ws_ping_interval = 0 to disable pings.
        """
    return self.settings.get('ws_ping_interval', WS_PING_INTERVAL)