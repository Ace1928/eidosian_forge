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
def on_reconnection(new_channel):
    """Callback invoked when the kombu reconnects and creates
            a new channel, we use it the reconfigure our consumers.
            """
    self._set_current_channel(new_channel)
    self.set_transport_socket_timeout()
    LOG.info('[%(connection_id)s] Reconnected to AMQP server on %(hostname)s:%(port)s via [%(transport)s] client with port %(client_port)s.', self._get_connection_info())