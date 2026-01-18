import queue as _queue
import struct as _struct
import socket as _socket
import asyncio as _asyncio
import threading as _threading
from pyglet.event import EventDispatcher as _EventDispatcher
from pyglet.util import debug_print
def on_connection(self, connection):
    """Event for new Client connections."""
    assert _debug_net(f'Connected <--- {connection}')
    connection.set_handler('on_disconnect', self.on_disconnect)