import unittest
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
def create_foo_container(self):
    instance_trait = self.trait_under_test

    class FooContainer(HasTraits):
        not_adapting_foo = instance_trait(Foo)
        adapting_foo = instance_trait(Foo, adapt='yes')
        adapting_foo_permissive = instance_trait(Foo, adapt='default')
        adapting_foo_dynamic_default = instance_trait(Foo, adapt='default', factory=default_foo)
        not_adapting_foo_list = List(Foo)
        adapting_foo_list = List(instance_trait(Foo, adapt='yes'))
    return FooContainer()