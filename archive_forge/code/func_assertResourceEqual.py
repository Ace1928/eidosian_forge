import uuid
from osc_placement.tests.functional import base
def assertResourceEqual(self, r1, r2):
    self.assertEqual(sorted_resources(r1), sorted_resources(r2))