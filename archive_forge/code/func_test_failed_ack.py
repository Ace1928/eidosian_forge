from taskflow.engines.worker_based import dispatcher
from taskflow import test
from taskflow.test import mock
def test_failed_ack(self):
    on_hello = mock.MagicMock()
    handlers = {'hello': dispatcher.Handler(on_hello)}
    d = dispatcher.TypeDispatcher(type_handlers=handlers)
    msg = mock_acked_message(ack_ok=False, properties={'type': 'hello'})
    d.on_message('', msg)
    self.assertTrue(msg.ack_log_error.called)
    self.assertFalse(msg.acknowledged)
    self.assertFalse(on_hello.called)