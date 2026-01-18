from oslo_config import cfg
from heat.common import config
from heat.common import crypt
from heat.common import exception
from heat.tests import common
def test_init_auth_encryption_key_length(self):
    """Test for length of the auth_encryption_length in config file"""
    cfg.CONF.set_override('auth_encryption_key', 'abcdefghijklma')
    err = self.assertRaises(exception.Error, config.startup_sanity_check)
    exp_msg = 'heat.conf misconfigured, auth_encryption_key must be 32 characters'
    self.assertIn(exp_msg, str(err))