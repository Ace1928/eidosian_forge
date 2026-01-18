import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def mock_notifier_exchange(name):

    def side_effect(target, ctxt, message, version, retry):
        target.exchange = name
        return transport._driver.send_notification(target, ctxt, message, version, retry=retry)
    transport._send_notification = mock.MagicMock(side_effect=side_effect)