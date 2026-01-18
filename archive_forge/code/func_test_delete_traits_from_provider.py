import uuid
from osc_placement.tests.functional import base
def test_delete_traits_from_provider(self):
    self.trait_create(TRAIT)
    rp = self.resource_provider_create()
    self.resource_provider_trait_set(rp['uuid'], TRAIT)
    traits = {t['name'] for t in self.resource_provider_trait_list(rp['uuid'])}
    self.assertEqual(1, len(traits))
    self.assertIn(TRAIT, traits)
    self.resource_provider_trait_delete(rp['uuid'])
    traits = {t['name'] for t in self.resource_provider_trait_list(rp['uuid'])}
    self.assertEqual(0, len(traits))