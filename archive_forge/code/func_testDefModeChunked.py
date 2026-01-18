import sys
from tests.base import BaseTestCase
from pyasn1.codec.der import decoder
from pyasn1.compat.octets import ints2octs, null
from pyasn1.error import PyAsn1Error
def testDefModeChunked(self):
    try:
        assert decoder.decode(ints2octs((35, 8, 3, 2, 0, 169, 3, 2, 1, 138)))
    except PyAsn1Error:
        pass
    else:
        assert 0, 'chunked encoding tolerated'