import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.der import encoder
from pyasn1.compat.octets import ints2octs
def testComponentsOrdering1(self):
    self.s.setComponentByName('status')
    self.s.getComponentByName('status').setComponentByPosition(0, 'A')
    assert encoder.encode(self.s) == ints2octs((49, 6, 2, 1, 5, 4, 1, 65))