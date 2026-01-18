import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
@unittest.skipUnless(six.PY3, 'Keyword only arguments Python 3 specific')
def test_create_autospec_keyword_only_arguments(self):
    func_def = 'def foo(a, *, b=None):\n    pass\n'
    namespace = {}
    exec(func_def, namespace)
    foo = namespace['foo']
    m = create_autospec(foo)
    m(1)
    m.assert_called_with(1)
    self.assertRaises(TypeError, m, 1, 2)
    m(2, b=3)
    m.assert_called_with(2, b=3)