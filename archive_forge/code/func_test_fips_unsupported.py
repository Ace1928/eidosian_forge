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
@mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.ensure_connection')
@mock.patch('oslo_messaging._drivers.impl_rabbit.ssl')
@mock.patch('kombu.connection.Connection')
def test_fips_unsupported(self, connection_klass, fake_ssl, fake_ensure):
    self.config(ssl=True, ssl_enforce_fips_mode=True, group='oslo_messaging_rabbit')
    transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
    self.addCleanup(transport.cleanup)
    del fake_ssl.FIPS_mode
    self.assertRaises(ConfigurationError, transport._driver._get_connection)