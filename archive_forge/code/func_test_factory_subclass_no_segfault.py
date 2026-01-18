import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_factory_subclass_no_segfault(self):
    """ Test that we can provide an instance as a default in the definition
        of a subclass.
        """
    obj = ConsumerSubclass()
    obj.x