from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_env_vars_security_token_beats_config_values(self):
    self.environ['AWS_ACCESS_KEY_ID'] = 'env_access_key'
    self.environ['AWS_SECRET_ACCESS_KEY'] = 'env_secret_key'
    self.environ['AWS_SECURITY_TOKEN'] = 'env_security_token'
    self.shared_config = {'default': {'aws_access_key_id': 'shared_access_key', 'aws_secret_access_key': 'shared_secret_key', 'aws_security_token': 'shared_security_token'}}
    self.config = {'Credentials': {'aws_access_key_id': 'cfg_access_key', 'aws_secret_access_key': 'cfg_secret_key', 'aws_security_token': 'cfg_security_token'}}
    p = provider.Provider('aws')
    self.assertEqual(p.access_key, 'env_access_key')
    self.assertEqual(p.secret_key, 'env_secret_key')
    self.assertEqual(p.security_token, 'env_security_token')
    self.environ.clear()
    p = provider.Provider('aws')
    self.assertEqual(p.security_token, 'shared_security_token')
    self.shared_config.clear()
    p = provider.Provider('aws')
    self.assertEqual(p.security_token, 'cfg_security_token')