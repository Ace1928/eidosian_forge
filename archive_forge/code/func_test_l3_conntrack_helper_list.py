import uuid
from openstackclient.tests.functional.network.v2 import common
def test_l3_conntrack_helper_list(self):
    helpers = [{'helper': 'tftp', 'protocol': 'udp', 'port': 69}, {'helper': 'ftp', 'protocol': 'tcp', 'port': 21}]
    expected_helpers = [{'Helper': 'tftp', 'Protocol': 'udp', 'Port': 69}, {'Helper': 'ftp', 'Protocol': 'tcp', 'Port': 21}]
    router_id = self._create_router()
    self._create_helpers(router_id, helpers)
    output = self.openstack('network l3 conntrack helper list %s ' % router_id, parse_output=True)
    for ct in output:
        self.assertEqual(router_id, ct.pop('Router ID'))
        ct.pop('ID')
        self.assertIn(ct, expected_helpers)