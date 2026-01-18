import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_autospec_property(self):

    class Foo(object):

        @property
        def foo(self):
            return 3
    foo = create_autospec(Foo)
    mock_property = foo.foo
    self.assertIsInstance(mock_property, MagicMock)
    mock_property(1, 2, 3)
    mock_property.abc(4, 5, 6)
    mock_property.assert_called_once_with(1, 2, 3)
    mock_property.abc.assert_called_once_with(4, 5, 6)