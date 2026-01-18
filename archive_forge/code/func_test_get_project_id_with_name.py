from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
@mock.patch.object(keystone.KeystoneClientPlugin, 'client')
def test_get_project_id_with_name(self, client_keystone):
    self._client.client.projects.get.side_effect = keystone_exceptions.NotFound
    self._client.client.projects.list.return_value = [self._get_mock_project()]
    client_keystone.return_value = self._client
    client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
    self.assertEqual(self.sample_uuid, client_plugin.get_project_id(self.sample_name))
    self.assertRaises(keystone_exceptions.NotFound, self._client.client.projects.get, self.sample_name)
    self._client.client.projects.list.assert_called_once_with(domain=None, name=self.sample_name)