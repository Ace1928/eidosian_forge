import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_create_no_resource_class(self):
    """Check errors on missing resource class."""
    base_cmd = 'baremetal allocation create'
    self.assertRaisesRegex(exceptions.CommandFailed, '--resource-class', self.openstack, base_cmd)