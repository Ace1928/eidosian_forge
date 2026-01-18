from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_metadata_server_returns_empty_string(self):
    self.get_instance_metadata.return_value = {'rolename': ''}
    with self.assertRaises(InvalidInstanceMetadataError):
        p = provider.Provider('aws')