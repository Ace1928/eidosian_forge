import uuid
from osc_placement.tests.functional import base
def test_delete_trait(self):
    self.trait_create(TRAIT)
    self.trait_delete(TRAIT)
    self.assertCommandFailed('HTTP 404', self.trait_show, TRAIT)