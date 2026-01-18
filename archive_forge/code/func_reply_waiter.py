import datetime
import ssl
import sys
import threading
import time
import uuid
import fixtures
import kombu
import kombu.connection
import kombu.transport.memory
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging._drivers import amqpdriver
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers import impl_rabbit as rabbit_driver
from oslo_messaging.exceptions import ConfigurationError
from oslo_messaging.exceptions import MessageDeliveryFailure
from oslo_messaging.tests import utils as test_utils
from oslo_messaging.transport import DriverLoadFailure
from unittest import mock
def reply_waiter(self, msg_id, timeout, call_monitor_timeout, reply_q):
    if wait_conditions:
        cond = wait_conditions.pop()
        with cond:
            cond.notify()
        with cond:
            cond.wait()
    return orig_reply_waiter(self, msg_id, timeout, call_monitor_timeout, reply_q)