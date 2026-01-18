import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_create_autospec_keyword_arguments(self):

    class Foo(object):
        a = 3
    m = create_autospec(Foo, a='3')
    self.assertEqual(m.a, '3')