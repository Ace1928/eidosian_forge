import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_nets_with_network(self):
    nets = [' network = 1234567 , v4-fixed-ip = 172.17.0.3 ']
    result = utils.parse_nets(nets)
    self.assertEqual([{'network': '1234567', 'v4-fixed-ip': '172.17.0.3'}], result)