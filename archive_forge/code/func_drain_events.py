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
def drain_events(self, connection, timeout=None):
    time_start = monotonic()
    get = self.cycle.get
    polling_interval = self.polling_interval
    if timeout and polling_interval and (polling_interval > timeout):
        polling_interval = timeout
    while 1:
        try:
            get(self._deliver, timeout=timeout)
        except Empty:
            if timeout is not None and monotonic() - time_start >= timeout:
                raise socket.timeout()
            if polling_interval is not None:
                sleep(polling_interval)
        else:
            break