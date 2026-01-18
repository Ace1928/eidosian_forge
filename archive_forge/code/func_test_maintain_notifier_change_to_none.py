import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_maintain_notifier_change_to_none(self):

    class UnassumingObserver(DummyObserver):

        def iter_observables(self, object):
            if object is None:
                raise ValueError('This observer cannot handle None.')
            yield from ()
    foo = ClassWithInstance()
    graph = create_graph(create_observer(name='instance', notify=True), UnassumingObserver())
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler)
    foo.instance = ClassWithTwoValue()
    try:
        foo.instance = None
    except Exception:
        self.fail('Setting instance back to None should not fail.')