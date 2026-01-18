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
def testSubtypeSpec(self):
    s = self.s1.clone(subtypeSpec=constraint.ConstraintsUnion(constraint.SingleValueConstraint(str2octs('abc'))))
    try:
        s.setComponentByPosition(0, univ.OctetString('abc'))
    except PyAsn1Error:
        assert 0, 'constraint fails'
    try:
        s.setComponentByPosition(1, univ.OctetString('Abc'))
    except PyAsn1Error:
        try:
            s.setComponentByPosition(1, univ.OctetString('Abc'), verifyConstraints=False)
        except PyAsn1Error:
            assert 0, 'constraint failes with verifyConstraints=True'
    else:
        assert 0, 'constraint fails'