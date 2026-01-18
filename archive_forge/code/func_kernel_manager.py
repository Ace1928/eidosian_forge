import json
import struct
from typing import Any, List
from jupyter_client.session import Session
from tornado.websocket import WebSocketHandler
from traitlets import Float, Instance, Unicode, default
from traitlets.config import LoggingConfigurable
from jupyter_client.jsonutil import extract_dates
from jupyter_server.transutils import _i18n
from .abc import KernelWebsocketConnectionABC
@property
def kernel_manager(self):
    """The kernel manager."""
    return self.parent