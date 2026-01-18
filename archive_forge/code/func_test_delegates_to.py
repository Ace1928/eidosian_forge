import gc
import time
import unittest
from traits.api import HasTraits, Any, DelegatesTo, Instance, Int
def test_delegates_to(self):
    """ Tests if an object that delegates to another is freed.
        """

    class Base(HasTraits):
        """ Object we are delegating to. """
        i = Int

    class Delegates(HasTraits):
        """ Object that delegates. """
        b = Instance(Base)
        i = DelegatesTo('b')
    b = Base()
    d = Delegates(b=b)
    del d
    for i in range(3):
        gc.collect(2)
    ds = [obj for obj in gc.get_objects() if isinstance(obj, Delegates)]
    self.assertEqual(ds, [])