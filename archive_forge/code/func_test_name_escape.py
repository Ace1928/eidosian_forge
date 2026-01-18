from heat.common import identifier
from heat.tests import common
def test_name_escape(self):
    hi = identifier.HeatIdentifier('t', ':%', 'i')
    self.assertEqual(':%', hi.stack_name)
    self.assertEqual('t/stacks/%3A%25/i', hi.url_path())
    self.assertEqual('arn:openstack:heat::t:stacks/%3A%25/i', hi.arn())