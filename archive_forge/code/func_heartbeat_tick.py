import logging
import socket
import uuid
import warnings
from array import array
from time import monotonic
from vine import ensure_promise
from . import __version__, sasl, spec
from .abstract_channel import AbstractChannel
from .channel import Channel
from .exceptions import (AMQPDeprecationWarning, ChannelError, ConnectionError,
from .method_framing import frame_handler, frame_writer
from .transport import Transport
def heartbeat_tick(self, rate=2):
    """Send heartbeat packets if necessary.

        Raises:
            ~amqp.exceptions.ConnectionForvced: if none have been
                received recently.

        Note:
            This should be called frequently, on the order of
            once per second.

        Keyword Arguments:
            rate (int): Number of heartbeat frames to send during the heartbeat
                        timeout
        """
    AMQP_HEARTBEAT_LOGGER.debug('heartbeat_tick : for connection %s', self._connection_id)
    if not self.heartbeat:
        return
    if rate <= 0:
        rate = 2
    sent_now = self.bytes_sent
    recv_now = self.bytes_recv
    if self.prev_sent is None or self.prev_sent != sent_now:
        self.last_heartbeat_sent = monotonic()
    if self.prev_recv is None or self.prev_recv != recv_now:
        self.last_heartbeat_received = monotonic()
    now = monotonic()
    AMQP_HEARTBEAT_LOGGER.debug('heartbeat_tick : Prev sent/recv: %s/%s, now - %s/%s, monotonic - %s, last_heartbeat_sent - %s, heartbeat int. - %s for connection %s', self.prev_sent, self.prev_recv, sent_now, recv_now, now, self.last_heartbeat_sent, self.heartbeat, self._connection_id)
    self.prev_sent, self.prev_recv = (sent_now, recv_now)
    if now > self.last_heartbeat_sent + self.heartbeat / rate:
        AMQP_HEARTBEAT_LOGGER.debug('heartbeat_tick: sending heartbeat for connection %s', self._connection_id)
        self.send_heartbeat()
        self.last_heartbeat_sent = monotonic()
    two_heartbeats = 2 * self.heartbeat
    two_heartbeats_interval = self.last_heartbeat_received + two_heartbeats
    heartbeats_missed = two_heartbeats_interval < monotonic()
    if self.last_heartbeat_received and heartbeats_missed:
        raise ConnectionForced('Too many heartbeats missed')