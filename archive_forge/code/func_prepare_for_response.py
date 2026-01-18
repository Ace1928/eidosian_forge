import abc
import collections
import logging
import os
import platform
import queue
import random
import sys
import threading
import time
import uuid
from oslo_utils import eventletutils
import proton
import pyngus
from oslo_messaging._drivers.amqp1_driver.addressing import AddresserFactory
from oslo_messaging._drivers.amqp1_driver.addressing import keyify
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_NOTIFY
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_RPC
from oslo_messaging._drivers.amqp1_driver import eventloop
from oslo_messaging import exceptions
from oslo_messaging.target import Target
from oslo_messaging import transport
def prepare_for_response(self, request, callback):
    """Apply a unique message identifier to this request message. This will
        be used to identify messages received in reply.  The identifier is
        placed in the 'id' field of the request message.  It is expected that
        the identifier will appear in the 'correlation-id' field of the
        corresponding response message.

        When the caller is done receiving replies, it must call cancel_response
        """
    request.id = uuid.uuid4().hex
    self._correlation[request.id] = callback
    request.reply_to = self._receiver.source_address
    return request.id