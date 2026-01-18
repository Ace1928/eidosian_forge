import unittest
from traits.api import (
def test_list_inside_union_default(self):

    class HasUnionWithList(HasTraits):
        foo = Union(List(Int), Str)
    has_union = HasUnionWithList()
    value = has_union.foo
    self.assertIsInstance(value, list)
    with self.assertRaises(TraitError):
        value.append('not an integer')