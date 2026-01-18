from unittest import mock
import uuid
from saharaclient.api import base as sahara_base
from heat.common import exception
from heat.engine.clients.os import sahara
from heat.tests import common
from heat.tests import utils
def test_get_plugin_id(self):
    """Tests the get_plugin_id function."""
    plugin_name = 'myfakeplugin'
    self.my_plugin.name = plugin_name

    def side_effect(name):
        if name == plugin_name:
            return self.my_plugin
        else:
            raise sahara_base.APIException(error_code=404, error_name='NOT_FOUND')
    self.sahara_client.plugins.get.side_effect = side_effect
    self.assertIsNone(self.sahara_plugin.get_plugin_id(plugin_name))
    self.assertRaises(exception.EntityNotFound, self.sahara_plugin.get_plugin_id, 'noplugin')
    calls = [mock.call(plugin_name), mock.call('noplugin')]
    self.sahara_client.plugins.get.assert_has_calls(calls)