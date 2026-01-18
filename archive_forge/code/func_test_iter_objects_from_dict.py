import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
def test_iter_objects_from_dict(self):
    instance = ClassWithDict()
    instance.values = {'1': 1, '2': 2}
    observer = create_observer()
    actual = list(observer.iter_objects(instance.values))
    self.assertCountEqual(actual, [1, 2])