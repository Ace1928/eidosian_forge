import collections
import contextlib
import errno
import functools
import itertools
import math
import os
import random
import socket
import ssl
import sys
import threading
import time
from urllib import parse
import uuid
from amqp import exceptions as amqp_ex
import kombu
import kombu.connection
import kombu.entity
import kombu.messaging
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import amqp as rpc_amqp
from oslo_messaging._drivers import amqpdriver
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as rpc_common
from oslo_messaging._drivers import pool
from oslo_messaging import _utils
from oslo_messaging import exceptions
def set_transport_socket_timeout(self, timeout=None):
    heartbeat_timeout = self.heartbeat_timeout_threshold
    if self._heartbeat_supported_and_enabled():
        if timeout is None:
            timeout = heartbeat_timeout
        else:
            timeout = min(heartbeat_timeout, timeout)
    try:
        sock = self.channel.connection.sock
    except AttributeError as e:
        LOG.debug('[%s] Failed to get socket attribute: %s' % (self.connection_id, str(e)))
    else:
        sock.settimeout(timeout)
        if sys.platform != 'win32' and sys.platform != 'darwin':
            try:
                timeout = timeout * 1000 if timeout is not None else 0
                sock.setsockopt(socket.IPPROTO_TCP, TCP_USER_TIMEOUT, int(math.ceil(timeout)))
            except socket.error as error:
                code = error[0]
                if code != errno.ENOPROTOOPT:
                    raise