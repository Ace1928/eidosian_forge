from unittest import mock
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
import stevedore
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.loading._plugins.identity import v2
from keystoneauth1.loading._plugins.identity import v3
from keystoneauth1.tests.unit.loading import utils
def test_loading_invalid_plugin(self):
    auth_type = uuid.uuid4().hex
    self.conf_fixture.config(auth_type=auth_type, group=self.GROUP)
    e = self.assertRaises(exceptions.NoMatchingPlugin, loading.load_auth_from_conf_options, self.conf_fixture.conf, self.GROUP)
    self.assertEqual(auth_type, e.name)