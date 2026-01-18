from heat.common import identifier
from heat.tests import common
def test_arn_parse_upper(self):
    arn = 'ARN:openstack:heat::t:stacks/s/i/p'
    hi = identifier.HeatIdentifier.from_arn(arn)
    self.assertEqual('s', hi.stack_name)
    self.assertEqual('i', hi.stack_id)
    self.assertEqual('/p', hi.path)