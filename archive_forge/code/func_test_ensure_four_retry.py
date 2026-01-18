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
def test_ensure_four_retry(self):
    mock_callback = mock.Mock(side_effect=IOError)
    self.assertRaises(oslo_messaging.MessageDeliveryFailure, self.connection.ensure, mock_callback, retry=4)
    expected = 5
    if kombu.VERSION < (5, 2, 4):
        expected = 6
    self.assertEqual(expected, mock_callback.call_count)