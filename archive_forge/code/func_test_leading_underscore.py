import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def test_leading_underscore(self):
    """ Test the name-mangling for the double-underscored change handlers.
        """
    obj = _LeadingUnderscore(_ok=2.0)
    obj._ok = 3.0
    self.assertEqual(obj.calls, [('_ok', 0.0, 2.0), ('_ok', 2.0, 3.0)])