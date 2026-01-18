import sys
from tests.base import BaseTestCase
from pyasn1.compat import binary
def test_bin_allones(self):
    assert '0b1111111111111111111111111111111111111111111111111111111111111111' == binary.bin(18446744073709551615)