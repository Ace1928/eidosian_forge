import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_no_target_topic(self):
    transport = msg_notifier.get_notification_transport(self.conf, url='fake:')
    listener = oslo_messaging.get_notification_listener(transport, [oslo_messaging.Target()], [mock.Mock()])
    try:
        listener.start()
    except Exception as ex:
        self.assertIsInstance(ex, oslo_messaging.InvalidTarget, ex)
    else:
        self.assertTrue(False)