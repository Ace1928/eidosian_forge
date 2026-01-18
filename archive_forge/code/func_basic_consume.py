from __future__ import annotations
import base64
import socket
import sys
import warnings
from array import array
from collections import OrderedDict, defaultdict, namedtuple
from itertools import count
from multiprocessing.util import Finalize
from queue import Empty
from time import monotonic, sleep
from typing import TYPE_CHECKING
from amqp.protocol import queue_declare_ok_t
from kombu.exceptions import ChannelError, ResourceError
from kombu.log import get_logger
from kombu.transport import base
from kombu.utils.div import emergency_dump_state
from kombu.utils.encoding import bytes_to_str, str_to_bytes
from kombu.utils.scheduling import FairCycle
from kombu.utils.uuid import uuid
from .exchange import STANDARD_EXCHANGE_TYPES
def basic_consume(self, queue, no_ack, callback, consumer_tag, **kwargs):
    """Consume from `queue`."""
    self._tag_to_queue[consumer_tag] = queue
    self._active_queues.append(queue)

    def _callback(raw_message):
        message = self.Message(raw_message, channel=self)
        if not no_ack:
            self.qos.append(message, message.delivery_tag)
        return callback(message)
    self.connection._callbacks[queue] = _callback
    self._consumers.add(consumer_tag)
    self._reset_cycle()