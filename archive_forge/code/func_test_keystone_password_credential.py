from castellan.common import exception
from castellan.common import utils
from castellan.tests import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
def test_keystone_password_credential(self):
    password_value = 'p4ssw0rd'
    self.config_fixture.config(auth_type='keystone_password', password=password_value, group='key_manager')
    ks_password_context = utils.credential_factory(conf=CONF)
    ks_password_context_class = ks_password_context.__class__.__name__
    self.assertEqual('KeystonePassword', ks_password_context_class)
    self.assertEqual(password_value, ks_password_context.password)