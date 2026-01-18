from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import job_templates as api_j
from saharaclient.osc.v2 import job_templates as osc_j
from saharaclient.tests.unit.osc.v1 import test_job_templates as tjt_v1
def test_job_template_create_minimum_options(self):
    arglist = ['--name', 'pig-job', '--type', 'Pig']
    verifylist = [('name', 'pig-job'), ('type', 'Pig')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.job_mock.create.assert_called_once_with(description=None, interface=None, is_protected=False, is_public=False, libs=None, mains=None, name='pig-job', type='Pig')