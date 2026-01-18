import unittest
from traits.api import Int, Float, String, HasRequiredTraits, TraitError
def test_missing_required_trait(self):
    with self.assertRaises(TraitError) as exc:
        RequiredTest(i_trait=3)
    self.assertEqual(exc.exception.args[0], 'The following required traits were not provided: f_trait, s_trait.')