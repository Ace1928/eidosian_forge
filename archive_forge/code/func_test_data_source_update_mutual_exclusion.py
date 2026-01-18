from osc_lib.tests import utils as osc_utils
import testtools
from unittest import mock
from saharaclient.api import data_sources as api_ds
from saharaclient.osc.v1 import data_sources as osc_ds
from saharaclient.tests.unit.osc.v1 import fakes
def test_data_source_update_mutual_exclusion(self):
    arglist = ['data-source', '--name', 'data-source', '--access-key', 'ak', '--secret-key', 'sk', '--url', 's3a://abc/def', '--password', 'pw']
    with testtools.ExpectedException(osc_utils.ParserException):
        self.check_parser(self.cmd, arglist, mock.Mock())