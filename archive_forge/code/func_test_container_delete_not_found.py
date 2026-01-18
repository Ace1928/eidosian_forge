import copy
from unittest import mock
from oslo_config import cfg
from zunclient import exceptions as zc_exc
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import zun
from heat.engine.resources.openstack.zun import container
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_container_delete_not_found(self):
    c = self._create_resource('container', self.rsrc_defn, self.stack)
    scheduler.TaskRunner(c.create)()
    c.client_plugin = mock.MagicMock()
    self.client.containers.delete.side_effect = Exception('Not Found')
    scheduler.TaskRunner(c.delete)()
    self.assertEqual((c.DELETE, c.COMPLETE), c.state)
    self.client.containers.delete.assert_called_once_with(c.resource_id, stop=True)
    mock_ignore_not_found = c.client_plugin.return_value.ignore_not_found
    self.assertEqual(1, mock_ignore_not_found.call_count)