from heat.common import identifier
from heat.tests import common
def test_arn_decode_escape_round_trip(self):
    arn = 'arn:openstack:heat::%3A%2F:stacks/%3A%25/%3A%2F/%3A/'
    hi = identifier.HeatIdentifier.from_arn(arn)
    self.assertEqual(arn, hi.arn())