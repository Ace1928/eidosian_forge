from heat.common import identifier
from heat.tests import common
def test_arn_url_parse_heat_invalid(self):
    url = self.url_prefix + 'arn:openstack:cool::t:stacks/s/i/p'
    self.assertRaises(ValueError, identifier.HeatIdentifier.from_arn_url, url)