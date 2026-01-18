import hashlib
import io
from unittest import mock
import uuid
import boto3
import botocore
from botocore import exceptions as boto_exceptions
from botocore import stub
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import s3
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
@mock.patch('glance_store.location.Location')
@mock.patch.object(boto3.session.Session, 'client')
def test_client_custom_region_name(self, mock_client, mock_loc):
    """Test a custom s3_store_region_name in config"""
    self.config(s3_store_host='http://example.com')
    self.config(s3_store_region_name='regionOne')
    self.config(s3_store_bucket_url_format='path')
    self.store.configure()
    mock_loc.accesskey = 'abcd'
    mock_loc.secretkey = 'efgh'
    mock_loc.bucket = 'bucket1'
    self.store._create_s3_client(mock_loc)
    mock_client.assert_called_with(config=mock.ANY, endpoint_url='http://example.com', region_name='regionOne', service_name='s3', use_ssl=False)