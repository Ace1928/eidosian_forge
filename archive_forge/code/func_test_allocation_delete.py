import uuid
from osc_placement.tests.functional import base
def test_allocation_delete(self):
    consumer_uuid = str(uuid.uuid4())
    self.resource_allocation_set(consumer_uuid, ['rp={},VCPU=2'.format(self.rp1['uuid']), 'rp={},MEMORY_MB=512'.format(self.rp1['uuid'])])
    self.assertTrue(self.resource_allocation_show(consumer_uuid))
    self.resource_allocation_delete(consumer_uuid)
    self.assertEqual([], self.resource_allocation_show(consumer_uuid))