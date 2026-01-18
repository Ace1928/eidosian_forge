import unittest
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
def test_list_trait_automatic_adaptation(self):
    bar = Bar()
    foo_container = self.create_foo_container()
    with self.assertRaises(TraitError):
        foo_container.not_adapting_foo_list = [bar]
    with self.assertRaises(TraitError):
        foo_container.adapting_foo_list = [bar]
    register_factory(bar_to_foo_adapter, Bar, Foo)
    with self.assertRaises(TraitError):
        foo_container.not_adapting_foo_list = [bar]
    foo_container.adapting_foo_list = [bar]
    self.assertIsInstance(foo_container.adapting_foo_list[0], Foo)