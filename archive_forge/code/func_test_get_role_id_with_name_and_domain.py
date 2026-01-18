from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
@mock.patch.object(keystone.KeystoneClientPlugin, 'client')
def test_get_role_id_with_name_and_domain(self, client_keystone):
    self._client.client.roles.get.side_effect = keystone_exceptions.NotFound
    self._client.client.roles.list.return_value = [self._get_mock_role()]
    client_keystone.return_value = self._client
    client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
    self.assertEqual(self.sample_uuid, client_plugin.get_role_id(self.sample_name_and_domain))
    self.assertRaises(keystone_exceptions.NotFound, self._client.client.roles.get, self.sample_name)
    self._client.client.roles.list.assert_called_once_with(domain=client_plugin.get_domain_id(self.sample_domain_uuid), name=self.sample_name)