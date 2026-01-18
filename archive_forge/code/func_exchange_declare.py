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
def exchange_declare(self, exchange=None, type='direct', durable=False, auto_delete=False, arguments=None, nowait=False, passive=False):
    """Declare exchange."""
    type = type or 'direct'
    exchange = exchange or 'amq.%s' % type
    if passive:
        if exchange not in self.state.exchanges:
            raise ChannelError('NOT_FOUND - no exchange {!r} in vhost {!r}'.format(exchange, self.connection.client.virtual_host or '/'), (50, 10), 'Channel.exchange_declare', '404')
        return
    try:
        prev = self.state.exchanges[exchange]
        if not self.typeof(exchange).equivalent(prev, exchange, type, durable, auto_delete, arguments):
            raise NotEquivalentError(NOT_EQUIVALENT_FMT.format(exchange, self.connection.client.virtual_host or '/'))
    except KeyError:
        self.state.exchanges[exchange] = {'type': type, 'durable': durable, 'auto_delete': auto_delete, 'arguments': arguments or {}, 'table': []}