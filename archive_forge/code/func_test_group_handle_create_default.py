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
def test_group_handle_create_default(self):
    values = {group.KeystoneGroup.NAME: None, group.KeystoneGroup.DESCRIPTION: self._get_property_schema_value_default(group.KeystoneGroup.DESCRIPTION), group.KeystoneGroup.DOMAIN: self._get_property_schema_value_default(group.KeystoneGroup.DOMAIN), group.KeystoneGroup.ROLES: None}

    def _side_effect(key):
        return values[key]
    mock_group = self._get_mock_group()
    self.groups.create.return_value = mock_group
    self.test_group.properties = mock.MagicMock()
    self.test_group.properties.get.side_effect = _side_effect
    self.test_group.properties.__getitem__.side_effect = _side_effect
    self.test_group.physical_resource_name = mock.MagicMock()
    self.test_group.physical_resource_name.return_value = 'foo'
    self.assertEqual(None, self.test_group.properties.get(group.KeystoneGroup.NAME))
    self.assertEqual('', self.test_group.properties.get(group.KeystoneGroup.DESCRIPTION))
    self.assertEqual('default', self.test_group.properties.get(group.KeystoneGroup.DOMAIN))
    self.test_group.handle_create()
    self.groups.create.assert_called_once_with(name='foo', description='', domain='default')