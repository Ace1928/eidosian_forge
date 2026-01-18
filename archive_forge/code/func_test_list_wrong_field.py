from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_list_wrong_field(self):
    command = 'baremetal node list --fields ABC'
    ex_text = 'invalid choice'
    self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)