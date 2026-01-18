import uuid
from openstackclient.tests.functional.network.v2 import common
def test_l3_conntrack_helper_create_and_delete(self):
    """Test create, delete multiple"""
    helpers = [{'helper': 'tftp', 'protocol': 'udp', 'port': 69}, {'helper': 'ftp', 'protocol': 'tcp', 'port': 21}]
    router_id = self._create_router()
    created_helpers = self._create_helpers(router_id, helpers)
    ct_ids = ' '.join([ct['id'] for ct in created_helpers])
    raw_output = self.openstack('--debug network l3 conntrack helper delete %(router)s %(ct_ids)s' % {'router': router_id, 'ct_ids': ct_ids})
    self.assertOutput('', raw_output)