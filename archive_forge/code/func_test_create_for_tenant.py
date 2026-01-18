from unittest import mock
from heat.engine.clients.os import zaqar
from heat.tests import common
from heat.tests import utils
def test_create_for_tenant(self):
    context = utils.dummy_context()
    plugin = context.clients.client_plugin('zaqar')
    client = plugin.create_for_tenant('other_tenant', 'token')
    self.assertEqual('other_tenant', client.conf['auth_opts']['options']['os_project_id'])
    self.assertEqual('token', client.conf['auth_opts']['options']['os_auth_token'])