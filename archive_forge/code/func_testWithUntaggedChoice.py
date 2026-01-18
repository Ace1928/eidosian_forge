import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testWithUntaggedChoice(self):
    c = univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('premium', univ.Boolean())))
    s = univ.Set(componentType=namedtype.NamedTypes(namedtype.NamedType('name', univ.OctetString()), namedtype.NamedType('customer', c)))
    s.setComponentByName('name', 'A')
    s.getComponentByName('customer').setComponentByName('premium', True)
    assert encoder.encode(s) == ints2octs((49, 128, 1, 1, 255, 4, 1, 65, 0, 0))