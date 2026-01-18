import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_batch_timeout(self):
    transport = oslo_messaging.get_notification_transport(self.conf, url='fake:')
    endpoint = mock.Mock()
    endpoint.info.return_value = None
    listener_thread = self._setup_listener(transport, [endpoint], batch=(5, 1))
    notifier = self._setup_notifier(transport)
    cxt = test_utils.TestContext()
    for _ in range(12):
        notifier.info(cxt, 'an_event.start', 'test message')
    self.wait_for_messages(3)
    self.assertFalse(listener_thread.stop())
    messages = [dict(ctxt=cxt, publisher_id='testpublisher', event_type='an_event.start', payload='test message', metadata={'message_id': mock.ANY, 'timestamp': mock.ANY})]
    endpoint.info.assert_has_calls([mock.call(messages * 5), mock.call(messages * 5), mock.call(messages * 2)])