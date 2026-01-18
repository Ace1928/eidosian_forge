import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_format_multiple_bad_args(self):
    args = ['K1=V1', 'K22.2.2.2']
    ex = self.assertRaises(exc.CommandError, utils.format_args, args)
    self.assertEqual('arguments must be a list of KEY=VALUE not K22.2.2.2', str(ex))