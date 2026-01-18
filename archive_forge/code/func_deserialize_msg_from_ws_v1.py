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
def deserialize_msg_from_ws_v1(ws_msg):
    """Deserialize a message using the v1 protocol."""
    offset_number = int.from_bytes(ws_msg[:8], 'little')
    offsets = [int.from_bytes(ws_msg[8 * (i + 1):8 * (i + 2)], 'little') for i in range(offset_number)]
    channel = ws_msg[offsets[0]:offsets[1]].decode('utf-8')
    msg_list = [ws_msg[offsets[i]:offsets[i + 1]] for i in range(1, offset_number - 1)]
    return (channel, msg_list)