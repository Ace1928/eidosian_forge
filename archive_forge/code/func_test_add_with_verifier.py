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
@mock.patch.object(boto3.session.Session, 'client')
def test_add_with_verifier(self, mock_client):
    """Assert 'verifier.update' is called when verifier is provided"""
    expected_image_id = str(uuid.uuid4())
    expected_s3_size = FIVE_KB
    expected_s3_contents = b'*' * expected_s3_size
    image_s3 = io.BytesIO(expected_s3_contents)
    fake_s3_client = botocore.session.get_session().create_client('s3')
    verifier = mock.MagicMock(name='mock_verifier')
    with stub.Stubber(fake_s3_client) as stubber:
        stubber.add_response(method='head_bucket', service_response={})
        stubber.add_client_error(method='head_object', service_error_code='404', service_message='')
        stubber.add_response(method='put_object', service_response={})
        mock_client.return_value = fake_s3_client
        self.store.add(expected_image_id, image_s3, expected_s3_size, self.hash_algo, verifier=verifier)
    verifier.update.assert_called_with(expected_s3_contents)