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
def test_get_common(self):
    opts = loading.get_auth_common_conf_options()
    for opt in opts:
        self.assertIsInstance(opt, cfg.Opt)
    self.assertEqual(2, len(opts))