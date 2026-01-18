import unittest
from unittest import mock
from traits.trait_types import Any, Dict, Event, Str, TraitDictObject
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_errors import TraitError
def test_items_set_to_false(self):

    class Foo(HasTraits):
        mapping = Dict(items=False)
    handler = mock.Mock()
    foo = Foo(mapping={})
    foo.on_trait_change(lambda: handler(), name='mapping_items')
    foo.mapping['1'] = 1
    handler.assert_not_called()