from unittest import mock
from openstackclient.common import extension
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_extension_list_identity(self):
    arglist = ['--identity']
    verifylist = [('identity', True)]
    datalist = ((self.identity_extension.name, self.identity_extension.alias, self.identity_extension.description),)
    self._test_extension_list_helper(arglist, verifylist, datalist)
    self.identity_extensions_mock.list.assert_called_with()