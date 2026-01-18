import uuid
from osc_placement.tests.functional import base
def test_list_traits_in(self):
    self.trait_create(TRAIT)
    rp = self.resource_provider_create()
    self.resource_provider_trait_set(rp['uuid'], TRAIT)
    traits = {t['name'] for t in self.trait_list(name='in:' + TRAIT)}
    self.assertEqual(1, len(traits))
    self.assertIn(TRAIT, traits)