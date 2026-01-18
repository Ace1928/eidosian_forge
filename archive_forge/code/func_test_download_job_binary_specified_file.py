from osc_lib.tests import utils as osc_u
import testtools
from unittest import mock
from saharaclient.api import job_binaries as api_jb
from saharaclient.osc.v1 import job_binaries as osc_jb
from saharaclient.tests.unit.osc.v1 import test_job_binaries as tjb_v1
def test_download_job_binary_specified_file(self):
    m_open = mock.mock_open()
    with mock.patch('builtins.open', m_open, create=True):
        arglist = ['job-binary', '--file', 'test']
        verifylist = [('job_binary', 'job-binary'), ('file', 'test')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.jb_mock.get_file.assert_called_once_with('jb_id')
        self.assertEqual('test', m_open.call_args[0][0])