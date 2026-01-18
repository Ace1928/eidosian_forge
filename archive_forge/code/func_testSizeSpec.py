import math
import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import error
from pyasn1.compat.octets import str2octs, ints2octs, octs2ints
from pyasn1.error import PyAsn1Error
def testSizeSpec(self):
    s = self.s1.clone(sizeSpec=constraint.ConstraintsUnion(constraint.ValueSizeConstraint(1, 1)))
    s.setComponentByPosition(0, univ.OctetString('abc'))
    try:
        s.verifySizeSpec()
    except PyAsn1Error:
        assert 0, 'size spec fails'
    s.setComponentByPosition(1, univ.OctetString('abc'))
    try:
        s.verifySizeSpec()
    except PyAsn1Error:
        pass
    else:
        assert 0, 'size spec fails'