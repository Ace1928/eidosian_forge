import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_any_and_datetime(self):
    mock = Mock()
    mock(datetime.now(), foo=datetime.now())
    mock.assert_called_with(ANY, foo=ANY)