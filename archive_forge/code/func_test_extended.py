import unittest
from traits.api import HasTraits, Instance, Int
def test_extended(self):
    """ Tests a dynamic trait change handler using extended names. """

    class Child(HasTraits):
        i = Int

    class Parent(HasTraits):
        child = Instance(Child)
    parent = Parent(child=Child())
    target = HasTraits()
    self.count = 0

    def count_notifies():
        self.count += 1
    parent.on_trait_change(count_notifies, 'child:i', target=target)
    parent.child.i = 10
    del target
    parent.child.i = 0
    self.assertEqual(self.count, 1)