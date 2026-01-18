import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def test_any_value_followed_by_list_items_old_bad_value(self):
    foo = Foo()
    foo.any_value = CannotCompare()
    handler = mock.Mock()
    observe(object=foo, expression=expression.trait('any_value').list_items(optional=True), handler=handler)
    foo.any_value = foo.list_of_int
    handler.reset_mock()
    foo.any_value.append(1)
    self.assertEqual(handler.call_count, 1)