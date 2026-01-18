import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def test_iter_objects_does_not_evaluate_default(self):
    foo = Foo()
    list(helpers.iter_objects(foo, 'int_with_default'))
    self.assertFalse(foo.int_with_default_computed, 'Expect the default not to have been computed.')