from unittest import mock
from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine.clients.os import neutron
from heat.engine.clients.os.neutron import lbaas_constraints as lc
from heat.engine.clients.os.neutron import neutron_constraints as nc
from heat.tests import common
from heat.tests import utils
def test_has_no_extension(self):
    mock_extensions = {'extensions': []}
    self.neutron_client.list_extensions.return_value = mock_extensions
    self.assertFalse(self.neutron_plugin.has_extension('lbaas'))