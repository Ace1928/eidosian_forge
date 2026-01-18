import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_nets_with_invalid_ip(self):
    nets = ['network=1234567, v4-fixed-ip=23.555.567,789']
    self.assertRaises(exc.CommandError, utils.parse_nets, nets)