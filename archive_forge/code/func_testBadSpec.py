import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import decoder
from pyasn1.error import PyAsn1Error
def testBadSpec(self):
    try:
        decoder.decode('', asn1Spec='not an Asn1Item')
    except PyAsn1Error:
        pass
    else:
        assert 0, 'Invalid asn1Spec accepted'