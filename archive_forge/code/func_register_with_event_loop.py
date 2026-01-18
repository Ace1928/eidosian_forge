from __future__ import annotations
import os
import select
import socket
import ssl
import sys
import uuid
from gettext import gettext as _
from queue import Empty
from time import monotonic
import amqp.protocol
from kombu.log import get_logger
from kombu.transport import base, virtual
from kombu.transport.virtual import Base64, Message
def register_with_event_loop(self, connection, loop):
    """Register a file descriptor and callback with the loop.

        Register the callback self.on_readable to be called when an
        external epoll loop sees that the file descriptor registered is
        ready for reading. The file descriptor is created by this Transport,
        and is written to when a message is available.

        Because supports_ev == True, Celery expects to call this method to
        give the Transport an opportunity to register a read file descriptor
        for external monitoring by celery using an Event I/O notification
        mechanism such as epoll. A callback is also registered that is to
        be called once the external epoll loop is ready to handle the epoll
        event associated with messages that are ready to be handled for
        this Transport.

        The registration call is made exactly once per Transport after the
        Transport is instantiated.

        :param connection: A reference to the connection associated with
            this Transport.
        :type connection: kombu.transport.qpid.Connection
        :param loop: A reference to the external loop.
        :type loop: kombu.asynchronous.hub.Hub

        """
    self.r, self._w = os.pipe()
    if fcntl is not None:
        fcntl.fcntl(self.r, fcntl.F_SETFL, os.O_NONBLOCK)
    self.use_async_interface = True
    loop.add_reader(self.r, self.on_readable, connection, loop)