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
def test_consume_from_missing_queue_with_io_error_on_redeclaration(self):
    transport = oslo_messaging.get_transport(self.conf, 'kombu+memory://')
    self.addCleanup(transport.cleanup)
    with transport._driver._get_connection(driver_common.PURPOSE_LISTEN) as conn:
        with mock.patch('kombu.Queue.consume') as consume, mock.patch('kombu.Queue.declare') as declare:
            conn.declare_topic_consumer(exchange_name='test', topic='test', callback=lambda msg: True)
            import amqp
            consume.side_effect = [amqp.NotFound, None]
            declare.side_effect = [IOError, None]
            conn.connection.connection.recoverable_connection_errors = (IOError,)
            conn.connection.connection.recoverable_channel_errors = ()
            self.assertEqual(1, declare.call_count)
            conn.connection.connection.drain_events = mock.Mock()
            conn.consume()
            self.assertEqual(3, declare.call_count)