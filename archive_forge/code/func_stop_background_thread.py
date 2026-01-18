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
def stop_background_thread(self):
    self._stopped.clear()
    self._should_terminate = True
    try:
        self._server_socket.close()
    except self._socket_error:
        pass
    if not self._stopped.is_set():
        temp_socket = socket.socket()
        temp_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if not temp_socket.connect_ex(self._sockname):
            temp_socket.close()
    self._stopped.wait()
    self._server_thread.join()