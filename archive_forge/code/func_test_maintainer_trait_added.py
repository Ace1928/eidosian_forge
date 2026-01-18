import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
def test_maintainer_trait_added(self):
    instance = DummyHasTraitsClass()
    notifier = DummyNotifier()
    maintainer = DummyNotifier()
    graph = create_graph(self.observer, DummyObserver(notify=True, notifier=notifier, maintainer=maintainer), DummyObserver())
    call_add_or_remove_notifiers(object=instance, handler=instance.dummy_method, target=instance, graph=graph, remove=False)
    instance.add_trait('good_name', Str())
    notifiers = instance._trait('good_name', 2)._notifiers(True)
    self.assertIn(notifier, notifiers)
    self.assertIn(maintainer, notifiers)
    instance.add_trait('bad_name', Str())
    notifiers = instance._trait('bad_name', 2)._notifiers(True)
    self.assertNotIn(notifier, notifiers)
    self.assertNotIn(maintainer, notifiers)