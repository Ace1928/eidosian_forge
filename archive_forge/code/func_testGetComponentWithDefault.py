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
def testGetComponentWithDefault(self):
    s = univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('name', univ.OctetString()), namedtype.NamedType('id', univ.Integer())))
    assert s.getComponentByPosition(0, default=None, instantiate=False) is None
    assert s.getComponentByPosition(1, default=None, instantiate=False) is None
    assert s.getComponentByName('name', default=None, instantiate=False) is None
    assert s.getComponentByName('id', default=None, instantiate=False) is None
    assert s.getComponentByType(univ.OctetString.tagSet, default=None) is None
    assert s.getComponentByType(univ.Integer.tagSet, default=None) is None
    s[1] = 123
    assert s.getComponentByPosition(1, default=None) is not None
    assert s.getComponentByPosition(1, univ.noValue) == 123
    s.clear()
    assert s.getComponentByPosition(1, default=None, instantiate=False) is None