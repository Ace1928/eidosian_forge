from tempest.lib import exceptions as tempest_exc
from openstackclient.tests.functional import base
def test_extension_show_not_exist(self):
    """Test extension show with not existed name"""
    if not self.haz_network:
        self.skipTest('No Network service present')
    name = 'not_existed_ext'
    try:
        self.openstack('extension show ' + name)
    except tempest_exc.CommandFailed as e:
        self.assertIn('No Extension found for', str(e))
        self.assertIn(name, str(e))
    else:
        self.fail('CommandFailed should be raised')