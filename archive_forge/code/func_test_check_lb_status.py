from unittest import mock
from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine.clients.os import neutron
from heat.engine.clients.os.neutron import lbaas_constraints as lc
from heat.engine.clients.os.neutron import neutron_constraints as nc
from heat.tests import common
from heat.tests import utils
def test_check_lb_status(self):
    self.neutron_client.show_loadbalancer.side_effect = [{'loadbalancer': {'provisioning_status': 'ACTIVE'}}, {'loadbalancer': {'provisioning_status': 'PENDING_CREATE'}}, {'loadbalancer': {'provisioning_status': 'ERROR'}}]
    self.assertTrue(self.neutron_plugin.check_lb_status('1234'))
    self.assertFalse(self.neutron_plugin.check_lb_status('1234'))
    self.assertRaises(exception.ResourceInError, self.neutron_plugin.check_lb_status, '1234')