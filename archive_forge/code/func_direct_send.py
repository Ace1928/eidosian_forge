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
def direct_send(self, msg_id, msg):
    """Send a 'direct' message."""
    exchange = kombu.entity.Exchange(name='', type='direct', durable=self.rabbit_transient_quorum_queue, auto_delete=True, passive=True)
    options = oslo_messaging.TransportOptions(at_least_once=self.direct_mandatory_flag)
    LOG.debug('Sending direct to %s', msg_id)
    self._ensure_publishing(self._publish_and_raises_on_missing_exchange, exchange, msg, routing_key=msg_id, transport_options=options)