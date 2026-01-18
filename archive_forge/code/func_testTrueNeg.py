import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import decoder
from pyasn1.error import PyAsn1Error
def testTrueNeg(self):
    assert decoder.decode(False, asn1Spec=univ.Boolean()) == univ.Boolean(False)