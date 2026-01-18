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
def test_cache_dictionary_config_builder_tls_disabled(self):
    """Validate the backend is reset to default if caching is disabled."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.pymemcache', tls_cafile='path_to_ca_file', tls_keyfile='path_to_key_file', tls_certfile='path_to_cert_file', tls_allowed_ciphers='allowed_ciphers')
    with mock.patch.object(ssl, 'create_default_context'):
        config_dict = cache._build_cache_config(self.config_fixture.conf)
        self.assertFalse(self.config_fixture.conf.cache.tls_enabled)
        ssl.create_default_context.assert_not_called()
        self.assertNotIn('test_prefix.arguments.tls_context', config_dict)