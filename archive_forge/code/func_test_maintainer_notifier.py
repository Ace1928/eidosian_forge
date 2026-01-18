import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
def test_maintainer_notifier(self):
    observer = create_observer(filter=lambda name, trait: type(trait.trait_type) is Instance)
    observable = DummyObservable()
    notifier = DummyNotifier()
    child_observer = WatchfulObserver(observables=[observable], notifier=notifier)
    instance = DummyParent()
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance, graph=create_graph(observer, child_observer), handler=handler)
    instance.instance = Dummy()
    self.assertEqual(observable.notifiers, [notifier])
    instance.instance = None
    self.assertEqual(observable.notifiers, [])