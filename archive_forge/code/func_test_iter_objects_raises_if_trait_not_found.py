import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_iter_objects_raises_if_trait_not_found(self):
    observer = create_observer(name='sally', optional=False)
    foo = ClassWithInstance()
    with self.assertRaises(ValueError) as e:
        next(observer.iter_objects(foo))
    self.assertEqual(str(e.exception), 'Trait named {!r} not found on {!r}.'.format('sally', foo))