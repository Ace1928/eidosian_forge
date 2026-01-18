import unittest
from unittest import mock
def test_call_chain_is_maintained(self):
    m = mock.Mock()
    m.test1().test2.test3().test4
    mock.seal(m)
    with self.assertRaises(AttributeError) as cm:
        m.test1().test2.test3().test4()
    self.assertIn('mock.test1().test2.test3().test4', str(cm.exception))