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
def declare_consumer(self, consumer):
    """Create a Consumer using the class that was passed in and
        add it to our list of consumers
        """

    def _connect_error(exc):
        log_info = {'topic': consumer.routing_key, 'err_str': exc}
        LOG.error("Failed to declare consumer for topic '%(topic)s': %(err_str)s", log_info)

    def _declare_consumer():
        consumer.declare(self)
        tag = self._active_tags.get(consumer.queue_name)
        if tag is None:
            tag = next(self._tags)
            self._active_tags[consumer.queue_name] = tag
            self._new_tags.add(tag)
        self._consumers[consumer] = tag
        return consumer
    with self._connection_lock:
        return self.ensure(_declare_consumer, error_callback=_connect_error)