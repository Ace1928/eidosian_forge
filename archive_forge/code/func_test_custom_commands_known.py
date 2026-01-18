import os
from testtools import matchers
from testtools import skipUnless
from pbr import testr_command
from pbr.tests import base
from pbr.tests import util
@skipUnless(testr_command.have_testr, 'testrepository not available')
def test_custom_commands_known(self):
    stdout, _, return_code = self.run_setup('--help-commands')
    self.assertFalse(return_code)
    self.assertThat(stdout, matchers.Contains(' testr '))