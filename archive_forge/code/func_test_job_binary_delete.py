from osc_lib.tests import utils as osc_u
import testtools
from unittest import mock
from saharaclient.api import job_binaries as api_jb
from saharaclient.osc.v1 import job_binaries as osc_jb
from saharaclient.tests.unit.osc.v1 import test_job_binaries as tjb_v1
def test_job_binary_delete(self):
    arglist = ['job-binary']
    verifylist = [('job_binary', ['job-binary'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.jb_mock.delete.assert_called_once_with('jb_id')