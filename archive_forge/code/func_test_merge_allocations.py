import uuid
from osc_lib import exceptions
from oslotest import base
from osc_placement.resources import allocation
def test_merge_allocations(self):
    rp1 = str(uuid.uuid4())
    allocations = ['rp={},VCPU=4,MEMORY_MB=16324'.format(rp1), 'rp={},VCPU=4,DISK_GB=4096'.format(rp1)]
    expected = {rp1: {'VCPU': 4, 'MEMORY_MB': 16324, 'DISK_GB': 4096}}
    self.assertEqual(expected, allocation.parse_allocations(allocations))