import unittest
from unittest import mock
from traits.trait_types import Any, Dict, Event, Str, TraitDictObject
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_errors import TraitError
def test_modified_event(self):

    class Foo(HasTraits):
        name = Str
        modified = Event

        @on_trait_change('name')
        def _fire_modified_event(self):
            self.modified = True

    class Bar(HasTraits):
        foos = Dict(Str, Foo)
        modified = Event

        @on_trait_change('foos_items,foos.modified')
        def _fire_modified_event(self, obj, trait_name, old, new):
            self.modified = True
    bar = Bar()
    listener = create_listener()
    bar.on_trait_change(listener, 'modified')
    bar.foos = {'dino': Foo(name='dino')}
    self.assertEqual(1, listener.called)
    self.assertEqual('modified', listener.trait_name)
    listener.initialize()
    fred = Foo(name='fred')
    bar.foos['fred'] = fred
    self.assertEqual(1, listener.called)
    self.assertEqual('modified', listener.trait_name)
    listener.initialize()
    fred.name = 'barney'
    self.assertEqual(1, listener.called)
    self.assertEqual('modified', listener.trait_name)
    listener.initialize()
    bar.foos['fred'] = Foo(name='wilma')
    self.assertEqual(1, listener.called)
    self.assertEqual('modified', listener.trait_name)