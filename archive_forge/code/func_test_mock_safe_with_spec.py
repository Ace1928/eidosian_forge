import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_mock_safe_with_spec(self):

    class Foo(object):

        def assert_bar(self):
            pass

        def assertSome(self):
            pass
    m = Mock(spec=Foo)
    m.assert_bar()
    m.assertSome()
    m.assert_bar.assert_called_once()
    m.assertSome.assert_called_once()