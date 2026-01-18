import uuid
from osc_placement.tests.functional import base
def test_allocation_show_not_found(self):
    consumer_uuid = str(uuid.uuid4())
    result = self.resource_allocation_show(consumer_uuid)
    self.assertEqual([], result)