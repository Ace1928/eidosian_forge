from osc_lib.tests import utils as osc_u
import testtools
from unittest import mock
from saharaclient.api import job_binaries as api_jb
from saharaclient.osc.v1 import job_binaries as osc_jb
from saharaclient.tests.unit.osc.v1 import test_job_binaries as tjb_v1
def test_job_binary_list_extra_search_opts(self):
    arglist = ['--name', 'bin']
    verifylist = [('name', 'bin')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['Name', 'Id', 'Url']
    self.assertEqual(expected_columns, columns)
    expected_data = [('job-binary', 'jb_id', 'swift://cont/test')]
    self.assertEqual(expected_data, list(data))