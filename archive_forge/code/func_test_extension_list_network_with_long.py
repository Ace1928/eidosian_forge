from unittest import mock
from openstackclient.common import extension
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_extension_list_network_with_long(self):
    arglist = ['--network', '--long']
    verifylist = [('network', True), ('long', True)]
    datalist = ((self.network_extension.name, self.network_extension.alias, self.network_extension.description, '', self.network_extension.updated_at, self.network_extension.links),)
    self._test_extension_list_helper(arglist, verifylist, datalist, long=True)
    self.network_client.extensions.assert_called_with()