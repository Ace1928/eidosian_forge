from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import jobs as api_j
from saharaclient.osc.v2 import jobs as osc_j
from saharaclient.tests.unit.osc.v1 import test_jobs as tj_v1
def test_job_update_nothing_updated(self):
    arglist = ['job_id']
    verifylist = [('job', 'job_id')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.j_mock.update.assert_called_once_with('job_id')