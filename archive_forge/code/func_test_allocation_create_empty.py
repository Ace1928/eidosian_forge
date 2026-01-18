import uuid
from osc_placement.tests.functional import base
def test_allocation_create_empty(self):
    consumer_uuid = str(uuid.uuid4())
    exc = self.assertRaises(base.CommandException, self.resource_allocation_set, consumer_uuid, [])
    self.assertIn('At least one resource allocation must be specified', str(exc))