from heat.common import identifier
from heat.tests import common
def test_arn_parse_stacks_invalid(self):
    arn = 'arn:openstack:heat::t:sticks/s/i'
    self.assertRaises(ValueError, identifier.HeatIdentifier.from_arn, arn)