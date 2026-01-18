from unittest import mock
from saharaclient.api import base
from saharaclient.tests.unit import base as test_base
def test_find_unique(self):
    expected = mock.Mock(test='foo')
    self.man.list = mock.MagicMock(return_value=[expected, mock.Mock(test='bar')])
    ex = self.assertRaises(base.APIException, self.man.find_unique, test='baz')
    self.assertEqual(404, ex.error_code)
    ex = self.assertRaises(base.APIException, self.man.find_unique)
    self.assertEqual(409, ex.error_code)
    self.assertEqual(expected, self.man.find_unique(test='foo'))