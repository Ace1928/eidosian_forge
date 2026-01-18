import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
def test_maintainer_keep_notify_flag(self):
    instance = DummyHasTraitsClass()
    notifier = DummyNotifier()
    graph = create_graph(self.observer, DummyObserver(notify=False, notifier=notifier))
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance, handler=handler, target=instance, graph=graph, remove=False)
    instance.add_trait('good_name', Str())
    notifiers = instance._trait('good_name', 2)._notifiers(True)
    self.assertNotIn(notifier, notifiers)