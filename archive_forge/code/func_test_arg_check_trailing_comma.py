import unittest
from traits.api import (
def test_arg_check_trailing_comma(self):
    ac = ArgCheckSimple(tc=self)
    with self.assertRaises(TraitError):
        ac.on_trait_change(ac.arg_check0, 'int1, int2,')