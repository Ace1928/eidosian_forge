from heat.common import identifier
from heat.tests import common
def test_id_decode(self):
    arn = 'arn:openstack:heat::t:stacks/s/%3A%2F'
    hi = identifier.HeatIdentifier.from_arn(arn)
    self.assertEqual(':/', hi.stack_id)