from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_no_credentials_provided(self):
    p = provider.Provider('aws', provider.NO_CREDENTIALS_PROVIDED, provider.NO_CREDENTIALS_PROVIDED, provider.NO_CREDENTIALS_PROVIDED)
    self.assertEqual(p.access_key, provider.NO_CREDENTIALS_PROVIDED)
    self.assertEqual(p.secret_key, provider.NO_CREDENTIALS_PROVIDED)
    self.assertEqual(p.security_token, provider.NO_CREDENTIALS_PROVIDED)