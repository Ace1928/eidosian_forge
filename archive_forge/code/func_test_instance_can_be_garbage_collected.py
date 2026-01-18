import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def test_instance_can_be_garbage_collected(self):
    instance = DummyClass()
    instance_ref = weakref.ref(instance)
    notifier = create_notifier(handler=instance.dummy_method)
    del instance
    self.assertIsNone(instance_ref())