from castellan.common import exception
from castellan.common import utils
from castellan.tests import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
def test_keystone_token_credential(self):
    token_value = 'ec9799cd921e4e0a8ab6111c08ebf065'
    self.config_fixture.config(auth_type='keystone_token', token=token_value, group='key_manager')
    ks_token_context = utils.credential_factory(conf=CONF)
    ks_token_context_class = ks_token_context.__class__.__name__
    self.assertEqual('KeystoneToken', ks_token_context_class)
    self.assertEqual(token_value, ks_token_context.token)