import errno
import os.path
import socket
import sys
import threading
import time
from ... import errors, trace
from ... import transport as _mod_transport
from ...hooks import Hooks
from ...i18n import gettext
from ...lazy_import import lazy_import
from breezy.bzr.smart import (
from breezy.transport import (
from breezy import (
def serve_conn(self, conn, thread_name_suffix):
    conn.setblocking(True)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    thread_name = 'smart-server-child' + thread_name_suffix
    handler = self._make_handler(conn)
    connection_thread = threading.Thread(None, handler.serve, name=thread_name, daemon=True)
    self._active_connections.append((handler, connection_thread))
    connection_thread.start()
    return connection_thread