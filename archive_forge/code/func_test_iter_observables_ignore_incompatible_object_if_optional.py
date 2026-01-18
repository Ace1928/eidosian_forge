import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
def test_iter_observables_ignore_incompatible_object_if_optional(self):
    observer = create_observer(optional=True)
    actual = list(observer.iter_observables(None))
    self.assertEqual(actual, [])