from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_metadata_server_returns_missing_keys(self):
    self.get_instance_metadata.return_value = {'allowall': {u'AccessKeyId': u'iam_access_key', u'Token': u'iam_token', u'Expiration': u'2012-09-01T03:57:34Z'}}
    with self.assertRaises(InvalidInstanceMetadataError):
        p = provider.Provider('aws')