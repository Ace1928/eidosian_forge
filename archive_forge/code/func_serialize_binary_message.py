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
def serialize_binary_message(msg):
    """serialize a message as a binary blob

    Header:

    4 bytes: number of msg parts (nbufs) as 32b int
    4 * nbufs bytes: offset for each buffer as integer as 32b int

    Offsets are from the start of the buffer, including the header.

    Returns
    -------
    The message serialized to bytes.

    """
    msg = msg.copy()
    buffers = list(msg.pop('buffers'))
    bmsg = json.dumps(msg, default=json_default).encode('utf8')
    buffers.insert(0, bmsg)
    nbufs = len(buffers)
    offsets = [4 * (nbufs + 1)]
    for buf in buffers[:-1]:
        offsets.append(offsets[-1] + len(buf))
    offsets_buf = struct.pack('!' + 'I' * (nbufs + 1), nbufs, *offsets)
    buffers.insert(0, offsets_buf)
    return b''.join(buffers)