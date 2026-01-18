from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import group
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_group_handle_create(self):
    mock_group = self._get_mock_group()
    self.groups.create.return_value = mock_group
    self.assertEqual('test_group_1', self.test_group.properties.get(group.KeystoneGroup.NAME))
    self.assertEqual('Test group', self.test_group.properties.get(group.KeystoneGroup.DESCRIPTION))
    self.assertEqual('default', self.test_group.properties.get(group.KeystoneGroup.DOMAIN))
    self.test_group.handle_create()
    self.groups.create.assert_called_once_with(name='test_group_1', description='Test group', domain='default')
    self.assertEqual(mock_group.id, self.test_group.resource_id)