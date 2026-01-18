import uuid
from osc_placement.tests.functional import base
def test_fail_show_unknown_trait(self):
    self.assertCommandFailed('HTTP 404', self.trait_show, 'UNKNOWN')