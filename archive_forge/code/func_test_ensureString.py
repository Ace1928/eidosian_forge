import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_ensureString(self):
    assert 'abc' == octets.ensureString('abc')
    assert '123' == octets.ensureString(123)