import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_format_args_parse_comma_false(self):
    li = utils.format_args(['K1=V1,K2=2.2.2.2,K=V'], parse_comma=False)
    self.assertEqual({'K1': 'V1,K2=2.2.2.2,K=V'}, li)