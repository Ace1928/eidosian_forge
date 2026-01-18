from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_config_profile_values_are_used(self):
    self.config = {'profile dev': {'aws_access_key_id': 'dev_access_key', 'aws_secret_access_key': 'dev_secret_key'}, 'profile prod': {'aws_access_key_id': 'prod_access_key', 'aws_secret_access_key': 'prod_secret_key'}, 'profile prod_withtoken': {'aws_access_key_id': 'prod_access_key', 'aws_secret_access_key': 'prod_secret_key', 'aws_security_token': 'prod_token'}, 'Credentials': {'aws_access_key_id': 'default_access_key', 'aws_secret_access_key': 'default_secret_key'}}
    p = provider.Provider('aws', profile_name='prod')
    self.assertEqual(p.access_key, 'prod_access_key')
    self.assertEqual(p.secret_key, 'prod_secret_key')
    p = provider.Provider('aws', profile_name='prod_withtoken')
    self.assertEqual(p.access_key, 'prod_access_key')
    self.assertEqual(p.secret_key, 'prod_secret_key')
    self.assertEqual(p.security_token, 'prod_token')
    q = provider.Provider('aws', profile_name='dev')
    self.assertEqual(q.access_key, 'dev_access_key')
    self.assertEqual(q.secret_key, 'dev_secret_key')