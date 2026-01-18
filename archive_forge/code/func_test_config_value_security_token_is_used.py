from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_config_value_security_token_is_used(self):
    self.config = {'Credentials': {'aws_access_key_id': 'cfg_access_key', 'aws_secret_access_key': 'cfg_secret_key', 'aws_security_token': 'cfg_security_token'}}
    p = provider.Provider('aws')
    self.assertEqual(p.access_key, 'cfg_access_key')
    self.assertEqual(p.secret_key, 'cfg_secret_key')
    self.assertEqual(p.security_token, 'cfg_security_token')