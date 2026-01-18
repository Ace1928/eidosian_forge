import unittest
from traits.observation._list_change_event import (
from traits.trait_list_object import TraitList
def test_list_change_event_repr(self):
    event = ListChangeEvent(object=[], index=3, removed=[1, 2], added=[3, 4])
    actual = repr(event)
    self.assertEqual(actual, 'ListChangeEvent(object=[], index=3, removed=[1, 2], added=[3, 4])')