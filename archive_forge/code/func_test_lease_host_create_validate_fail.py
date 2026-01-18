from unittest import mock
from blazarclient import exception as client_exception
from oslo_utils.fixture import uuidsentinel as uuids
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import blazar
from heat.engine.resources.openstack.blazar import lease
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_lease_host_create_validate_fail(self):
    self.patchobject(lease.Lease, 'client_plugin', return_value=self.client)
    self.client.has_host.return_value = False
    lease_resource = self._create_resource('lease', self.rsrc_defn, self.stack)
    self.assertEqual(self.lease['name'], lease_resource.properties.get(lease.Lease.NAME))
    self.assertRaises(exception.StackValidationFailed, lease_resource.validate)