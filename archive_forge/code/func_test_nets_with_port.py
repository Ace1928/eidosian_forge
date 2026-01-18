import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_nets_with_port(self):
    nets = ['port=1234567, v6-fixed-ip=2001:db8::2']
    result = utils.parse_nets(nets)
    self.assertEqual([{'port': '1234567', 'v6-fixed-ip': '2001:db8::2'}], result)