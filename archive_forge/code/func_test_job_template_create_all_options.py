from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import job_templates as api_j
from saharaclient.osc.v2 import job_templates as osc_j
from saharaclient.tests.unit.osc.v1 import test_job_templates as tjt_v1
def test_job_template_create_all_options(self):
    arglist = ['--name', 'pig-job', '--type', 'Pig', '--mains', 'main', '--libs', 'lib', '--description', 'descr', '--public', '--protected']
    verifylist = [('name', 'pig-job'), ('type', 'Pig'), ('mains', ['main']), ('libs', ['lib']), ('description', 'descr'), ('public', True), ('protected', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.job_mock.create.assert_called_once_with(description='descr', interface=None, is_protected=True, is_public=True, libs=['jb_id'], mains=['jb_id'], name='pig-job', type='Pig')
    expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Libs', 'Mains', 'Name', 'Type')
    self.assertEqual(expected_columns, columns)
    expected_data = ('Job for test', 'job_id', False, False, 'lib:lib_id', 'main:main_id', 'pig-job', 'Pig')
    self.assertEqual(expected_data, data)