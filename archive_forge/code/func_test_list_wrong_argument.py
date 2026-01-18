import ddt
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_list_wrong_argument(self):
    """Test for baremetal node list with wrong argument."""
    command = 'baremetal node list --wrong_arg'
    ex_text = 'error: unrecognized arguments: --wrong_arg'
    self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)