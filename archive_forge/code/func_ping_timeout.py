import re
import warnings
from typing import Optional, no_type_check
from urllib.parse import urlparse
from tornado import ioloop, web
from tornado.iostream import IOStream
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.utils import JupyterServerAuthWarning
@property
def ping_timeout(self):
    """If no ping is received in this many milliseconds,
        close the websocket connection (VPNs, etc. can fail to cleanly close ws connections).
        Default is max of 3 pings or 30 seconds.
        """
    return self.settings.get('ws_ping_timeout', max(3 * self.ping_interval, WS_PING_INTERVAL))