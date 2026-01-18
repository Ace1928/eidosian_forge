from heat.common import identifier
from heat.tests import common
def test_arn_parse_arn_invalid(self):
    arn = 'urn:openstack:heat::t:stacks/s/i'
    self.assertRaises(ValueError, identifier.HeatIdentifier.from_arn, arn)