import json
from unittest import mock
from keystoneauth1 import loading as ks_loading
from keystoneauth1 import session
from heat.common import auth_plugin
from heat.common import config
from heat.common import exception
from heat.common import policy
from heat.tests import common
def test_get_keystone_plugin_loader_with_no_AuthZ(self):
    self.m_plugin.load_from_options().get_token.side_effect = Exception
    self.assertRaises(exception.AuthorizationFailure, self._get_keystone_plugin_loader)
    self.assertEqual(1, self.m_plugin.load_from_options().get_token.call_count)