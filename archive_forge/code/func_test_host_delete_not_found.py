from unittest import mock
from blazarclient import exception as client_exception
from oslo_utils.fixture import uuidsentinel as uuids
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import blazar
from heat.engine.resources.openstack.blazar import host
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_host_delete_not_found(self):
    host_resource = self._create_resource('host', self.rsrc_defn, self.stack)
    scheduler.TaskRunner(host_resource.create)()
    self.client.host.delete.side_effect = client_exception.BlazarClientException(code=404)
    self.client.host.get.side_effect = client_exception.BlazarClientException(code=404)
    scheduler.TaskRunner(host_resource.delete)()
    self.assertEqual((host_resource.DELETE, host_resource.COMPLETE), host_resource.state)