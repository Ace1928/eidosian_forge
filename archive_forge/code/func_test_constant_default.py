import unittest
from traits.api import (
def test_constant_default(self):

    class HasUnionWithList(HasTraits):
        foo = Union(Int(23), Float)
        nested = Union(Union(Str(), Bytes()), Union(Int(), Float(), None))
    has_union = HasUnionWithList()
    value = has_union.foo
    self.assertEqual(value, 23)
    self.assertEqual(has_union.trait('foo').default_value(), (DefaultValue.constant, 23))
    self.assertEqual(has_union.trait('nested').default_value(), (DefaultValue.constant, ''))