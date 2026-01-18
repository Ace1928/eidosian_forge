from osc_lib.tests import utils as osc_u
import testtools
from unittest import mock
from saharaclient.api import job_binaries as api_jb
from saharaclient.osc.v1 import job_binaries as osc_jb
from saharaclient.tests.unit.osc.v1 import test_job_binaries as tjb_v1
def test_job_binary_update_mutual_exclusion(self):
    arglist = ['job-binary', '--name', 'job-binary', '--access-key', 'ak', '--secret-key', 'sk', '--url', 's3://abc/def', '--password', 'pw']
    with testtools.ExpectedException(osc_u.ParserException):
        self.check_parser(self.cmd, arglist, mock.Mock())