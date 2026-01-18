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
def declare_direct_consumer(self, topic, callback):
    """Create a 'direct' queue.
        In nova's use, this is generally a msg_id queue used for
        responses for call/multicall
        """
    consumer = Consumer(exchange_name='', queue_name=topic, routing_key='', type='direct', durable=self.rabbit_transient_quorum_queue, exchange_auto_delete=False, queue_auto_delete=False, callback=callback, rabbit_ha_queues=self.rabbit_ha_queues, rabbit_queue_ttl=self.rabbit_transient_queues_ttl, enable_cancel_on_failover=self.enable_cancel_on_failover, rabbit_quorum_queue=self.rabbit_transient_quorum_queue, rabbit_quorum_queue_config=self.rabbit_quorum_queue_config)
    self.declare_consumer(consumer)