import sys
from tests.base import BaseTestCase
from pyasn1.compat import binary
def test_bin_zero(self):
    assert '0b0' == binary.bin(0)