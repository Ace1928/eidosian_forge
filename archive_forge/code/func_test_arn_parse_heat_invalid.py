from heat.common import identifier
from heat.tests import common
def test_arn_parse_heat_invalid(self):
    arn = 'arn:openstack:cool::t:stacks/s/i'
    self.assertRaises(ValueError, identifier.HeatIdentifier.from_arn, arn)