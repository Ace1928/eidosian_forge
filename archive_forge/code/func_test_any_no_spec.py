import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_any_no_spec(self):

    class Foo:

        def __eq__(self, other):
            pass
    mock = Mock()
    mock(Foo(), 1)
    mock.assert_has_calls([call(ANY, 1)])
    mock.assert_called_with(ANY, 1)
    mock.assert_any_call(ANY, 1)