from unittest import mock
from openstackclient.common import extension
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_extension_list_compute_and_network(self):
    arglist = ['--compute', '--network']
    verifylist = [('compute', True), ('network', True)]
    datalist = ((self.compute_extension.name, self.compute_extension.alias, self.compute_extension.description), (self.network_extension.name, self.network_extension.alias, self.network_extension.description))
    self._test_extension_list_helper(arglist, verifylist, datalist)
    self.compute_extensions_mock.assert_called_with()
    self.network_client.extensions.assert_called_with()