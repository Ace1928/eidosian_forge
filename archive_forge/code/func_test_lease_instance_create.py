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
def test_lease_instance_create(self):
    t = template_format.parse(blazar_lease_instance_template)
    stack = utils.parse_stack(t)
    resource_defn = stack.t.resource_definitions(stack)
    rsrc_defn = resource_defn['test-lease']
    lease_resource = self._create_resource('lease', rsrc_defn, stack)
    self.assertEqual(self.lease['name'], lease_resource.properties.get(lease.Lease.NAME))
    scheduler.TaskRunner(lease_resource.create)()
    self.assertEqual(uuids.lease_id, lease_resource.resource_id)
    self.assertEqual((lease_resource.CREATE, lease_resource.COMPLETE), lease_resource.state)
    self.assertEqual('lease', lease_resource.entity)
    reservations = [{'resource_type': 'virtual:instance', 'amount': 1, 'vcpus': 1, 'memory_mb': 512, 'disk_gb': 15, 'affinity': False, 'resource_properties': ''}]
    self.client.lease.create.assert_called_once_with(name=self.lease['name'], start=self.lease['start_date'], end=self.lease['end_date'], reservations=reservations, events=self.lease['events'])