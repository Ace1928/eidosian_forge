import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
@mock.patch.object(logging, 'exception')
@mock.patch.object(timeutils, 'now')
def test_exc_retrier_same_10exc_2min_gives_2logs(self, mock_now, mock_log):
    self._exceptions = [Exception('unexpected 1')]
    mock_now_side_effect = [0]
    for ts in [12, 23, 34, 45]:
        self._exceptions.append(Exception('unexpected 1'))
        mock_now_side_effect.append(ts)
    self._exceptions.append(Exception('unexpected 1'))
    mock_now_side_effect.append(106)
    for ts in [106, 107]:
        mock_now_side_effect.append(ts)
    for ts in [117, 128, 139, 150]:
        self._exceptions.append(Exception('unexpected 1'))
        mock_now_side_effect.append(ts)
    mock_now.side_effect = mock_now_side_effect
    self.exception_generator()
    self.assertEqual([], self._exceptions)
    self.assertEqual(12, len(mock_now.mock_calls))
    self.assertEqual(2, len(mock_log.mock_calls))
    mock_log.assert_has_calls([mock.call('Unexpected exception occurred 1 time(s)... retrying.'), mock.call('Unexpected exception occurred 5 time(s)... retrying.')])