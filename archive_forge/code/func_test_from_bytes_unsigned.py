import sys
from tests.base import BaseTestCase
from pyasn1.compat import integer
def test_from_bytes_unsigned(self):
    assert -66051 == integer.from_bytes('þýý', signed=True)