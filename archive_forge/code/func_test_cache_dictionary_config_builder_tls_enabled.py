import copy
import ssl
import time
from unittest import mock
from dogpile.cache import proxy
from oslo_config import cfg
from oslo_utils import uuidutils
from pymemcache import KeepaliveOpts
from oslo_cache import _opts
from oslo_cache import core as cache
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_cache_dictionary_config_builder_tls_enabled(self):
    """Validate the backend is reset to default if caching is disabled."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.pymemcache', tls_enabled=True)
    fake_context = mock.Mock()
    with mock.patch.object(ssl, 'create_default_context', return_value=fake_context):
        config_dict = cache._build_cache_config(self.config_fixture.conf)
        self.assertTrue(self.config_fixture.conf.cache.tls_enabled)
        ssl.create_default_context.assert_called_with(cafile=None)
        fake_context.load_cert_chain.assert_not_called()
        fake_context.set_ciphers.assert_not_called()
        self.assertEqual(fake_context, config_dict['test_prefix.arguments.tls_context'])