import json
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
@ddt.data(('--uuid', '', 'expected one argument'), ('--uuid', '!@#$^*&%^', 'Expected UUID for uuid'), ('', '', 'the following arguments are required'), ('', 'not/a/name', 'does not match'), ('', 'foo', 'does not match'), ('--steps', '', 'expected one argument'), ('--steps', '[]', 'is too short'))
@ddt.unpack
def test_create_negative(self, argument, value, ex_text):
    """Check errors on invalid input parameters."""
    base_cmd = 'baremetal deploy template create'
    if argument != '':
        base_cmd += ' %s' % self._get_random_trait()
    if argument != '--steps':
        base_cmd += " --steps '%s'" % self.steps
    command = self.construct_cmd(base_cmd, argument, value)
    self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)