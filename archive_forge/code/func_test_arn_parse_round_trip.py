from heat.common import identifier
from heat.tests import common
def test_arn_parse_round_trip(self):
    arn = 'arn:openstack:heat::t:stacks/s/i/p'
    hi = identifier.HeatIdentifier.from_arn(arn)
    self.assertEqual(arn, hi.arn())