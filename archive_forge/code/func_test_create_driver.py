import ddt
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
@ddt.data(('', '', 'error: the following arguments are required: --driver'), ('--driver', 'wrongdriver', 'No valid host was found. Reason: No conductor service registered which supports driver wrongdriver.'))
@ddt.unpack
def test_create_driver(self, argument, value, ex_text):
    """Negative test for baremetal node driver options."""
    base_cmd = 'baremetal node create'
    command = self.construct_cmd(base_cmd, argument, value)
    self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)