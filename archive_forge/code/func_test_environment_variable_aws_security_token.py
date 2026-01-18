from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_environment_variable_aws_security_token(self):
    self.environ['AWS_ACCESS_KEY_ID'] = 'env_access_key'
    self.environ['AWS_SECRET_ACCESS_KEY'] = 'env_secret_key'
    self.environ['AWS_SECURITY_TOKEN'] = 'env_security_token'
    p = provider.Provider('aws')
    self.assertEqual(p.access_key, 'env_access_key')
    self.assertEqual(p.secret_key, 'env_secret_key')
    self.assertEqual(p.security_token, 'env_security_token')