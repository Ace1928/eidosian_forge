from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_metadata_server_credentials(self):
    self.get_instance_metadata.return_value = INSTANCE_CONFIG
    p = provider.Provider('aws')
    self.assertEqual(p.access_key, 'iam_access_key')
    self.assertEqual(p.secret_key, 'iam_secret_key')
    self.assertEqual(p.security_token, 'iam_token')
    self.assertEqual(self.get_instance_metadata.call_args[1]['data'], 'meta-data/iam/security-credentials/')