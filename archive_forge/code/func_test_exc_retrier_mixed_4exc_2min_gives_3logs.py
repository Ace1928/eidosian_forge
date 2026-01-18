import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
@mock.patch.object(logging, 'exception')
@mock.patch.object(timeutils, 'now')
def test_exc_retrier_mixed_4exc_2min_gives_3logs(self, mock_now, mock_log):
    self._exceptions = [Exception('unexpected 1')]
    mock_now_side_effect = [0]
    self._exceptions.append(Exception('unexpected 1'))
    mock_now_side_effect.append(10)
    self._exceptions.append(Exception('unexpected 1'))
    mock_now_side_effect.extend([100, 100, 105])
    self._exceptions.append(Exception('unexpected 2'))
    mock_now_side_effect.extend([110, 111])
    mock_now.side_effect = mock_now_side_effect
    self.exception_generator()
    self.assertEqual([], self._exceptions)
    self.assertEqual(7, len(mock_now.mock_calls))
    self.assertEqual(3, len(mock_log.mock_calls))
    mock_log.assert_has_calls([mock.call('Unexpected exception occurred 1 time(s)... retrying.'), mock.call('Unexpected exception occurred 2 time(s)... retrying.'), mock.call('Unexpected exception occurred 1 time(s)... retrying.')])