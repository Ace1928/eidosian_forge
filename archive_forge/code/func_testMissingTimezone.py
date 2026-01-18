import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testMissingTimezone(self):
    try:
        assert encoder.encode(useful.UTCTime('150501120112')) == ints2octs((23, 13, 49, 53, 48, 53, 48, 49, 49, 50, 48, 49, 49, 50, 90))
    except PyAsn1Error:
        pass
    else:
        assert 0, 'Missing timezone tolerated'