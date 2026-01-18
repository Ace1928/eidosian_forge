import uuid
from openstackclient.tests.functional.network.v2 import common
def test_ip_availability_list(self):
    """Test ip availability list"""
    cmd_output = self.openstack('ip availability list', parse_output=True)
    names = [x['Network Name'] for x in cmd_output]
    self.assertIn(self.NETWORK_NAME, names)