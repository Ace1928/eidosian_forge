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
def restore_unacked_once(self, stderr=None):
    """Restore all unacknowledged messages at shutdown/gc collect.

        Note:
        ----
            Can only be called once for each instance, subsequent
            calls will be ignored.
        """
    self._on_collect.cancel()
    self._flush()
    stderr = sys.stderr if stderr is None else stderr
    state = self._delivered
    if not self.restore_at_shutdown or not self.channel.do_restore:
        return
    if getattr(state, 'restored', None):
        assert not state
        return
    try:
        if state:
            print(RESTORING_FMT.format(len(self._delivered)), file=stderr)
            unrestored = self.restore_unacked()
            if unrestored:
                errors, messages = list(zip(*unrestored))
                print(RESTORE_PANIC_FMT.format(len(errors), errors), file=stderr)
                emergency_dump_state(messages, stderr=stderr)
    finally:
        state.restored = True