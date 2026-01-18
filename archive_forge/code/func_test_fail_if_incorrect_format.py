import uuid
from osc_lib import exceptions
from oslotest import base
from osc_placement.resources import allocation
def test_fail_if_incorrect_format(self):
    allocations = ['incorrect_format']
    self.assertRaisesRegex(ValueError, 'Incorrect allocation', allocation.parse_allocations, allocations)
    allocations = ['=,']
    self.assertRaisesRegex(ValueError, '2 is required', allocation.parse_allocations, allocations)
    allocations = ['abc=155']
    self.assertRaisesRegex(ValueError, 'Incorrect allocation', allocation.parse_allocations, allocations)
    allocations = ['abc=155,xyz=999']
    self.assertRaisesRegex(ValueError, 'parameter is required', allocation.parse_allocations, allocations)