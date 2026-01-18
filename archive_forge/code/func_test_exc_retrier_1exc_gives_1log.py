import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
@mock.patch.object(logging, 'exception')
@mock.patch.object(timeutils, 'now')
def test_exc_retrier_1exc_gives_1log(self, mock_now, mock_log):
    self._exceptions = [Exception('unexpected %d' % 1)]
    mock_now.side_effect = [0]
    self.exception_generator()
    self.assertEqual([], self._exceptions)
    mock_log.assert_called_once_with('Unexpected exception occurred %d time(s)... retrying.' % 1)
    mock_now.assert_has_calls([mock.call()])