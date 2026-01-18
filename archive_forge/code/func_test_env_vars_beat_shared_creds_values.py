from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_env_vars_beat_shared_creds_values(self):
    self.environ['AWS_ACCESS_KEY_ID'] = 'env_access_key'
    self.environ['AWS_SECRET_ACCESS_KEY'] = 'env_secret_key'
    self.shared_config = {'default': {'aws_access_key_id': 'shared_access_key', 'aws_secret_access_key': 'shared_secret_key'}}
    p = provider.Provider('aws')
    self.assertEqual(p.access_key, 'env_access_key')
    self.assertEqual(p.secret_key, 'env_secret_key')
    self.assertIsNone(p.security_token)