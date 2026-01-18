import logging
from threading import Thread, Lock, Event
import ncclient.transport
from ncclient.xml_ import *
from ncclient.capabilities import Capabilities
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport.errors import TransportError, SessionError, SessionCloseError
from ncclient.transport.notify import Notification
def get_listener_instance(self, cls):
    """If a listener of the specified type is registered, returns the
        instance.

        :type cls: :class:`SessionListener`
        """
    with self._lock:
        for listener in self._listeners:
            if isinstance(listener, cls):
                return listener