import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_nets_with_only_ip(self):
    nets = ['v4-fixed-ip = 172.17.0.3']
    self.assertRaises(exc.CommandError, utils.parse_nets, nets)