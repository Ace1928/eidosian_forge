import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.der import encoder
from pyasn1.compat.octets import ints2octs
def testUninitializedOptionalNullIsNotEncoded(self):
    self.s = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.OptionalNamedType('null', univ.Null())))
    self.s.clear()
    assert encoder.encode(self.s) == ints2octs((48, 0))