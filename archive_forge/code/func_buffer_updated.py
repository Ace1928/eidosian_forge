import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def buffer_updated(self, nbytes):
    self._incoming.write(self._ssl_buffer_view[:nbytes])
    if self._state == SSLProtocolState.DO_HANDSHAKE:
        self._do_handshake()
    elif self._state == SSLProtocolState.WRAPPED:
        self._do_read()
    elif self._state == SSLProtocolState.FLUSHING:
        self._do_flush()
    elif self._state == SSLProtocolState.SHUTDOWN:
        self._do_shutdown()