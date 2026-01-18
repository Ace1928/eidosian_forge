from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_config_values_are_used(self):
    self.config = {'Credentials': {'aws_access_key_id': 'cfg_access_key', 'aws_secret_access_key': 'cfg_secret_key'}}
    p = provider.Provider('aws')
    self.assertEqual(p.access_key, 'cfg_access_key')
    self.assertEqual(p.secret_key, 'cfg_secret_key')
    self.assertIsNone(p.security_token)