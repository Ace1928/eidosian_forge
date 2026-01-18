import logging
import fixtures
import oslo_messaging
from oslo_messaging.notify import log_handler
from oslo_messaging.tests import utils as test_utils
from unittest import mock
@mock.patch('oslo_messaging.notify.notifier.Notifier._notify')
def test_emit_notification(self, mock_notify):
    logrecord = logging.LogRecord(name='name', level='ERROR', pathname='/tmp', lineno=1, msg='Message', args=None, exc_info=None)
    self.publisherrorshandler.emit(logrecord)
    self.assertEqual('error.publisher', self.publisherrorshandler._notifier.publisher_id)
    mock_notify.assert_called_with({}, 'error_notification', {'error': 'Message'}, 'ERROR')