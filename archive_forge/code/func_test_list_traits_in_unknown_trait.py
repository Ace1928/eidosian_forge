import uuid
from osc_placement.tests.functional import base
def test_list_traits_in_unknown_trait(self):
    traits = {t['name'] for t in self.trait_list(name='in:CUSTOM_FOO')}
    self.assertEqual(0, len(traits))