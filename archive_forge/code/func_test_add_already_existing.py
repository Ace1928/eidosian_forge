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
def test_add_already_existing(self, mock_client):
    """Tests that adding an image with an existing identifier
        raises an appropriate exception
        """
    image_s3 = io.BytesIO(b'never_gonna_make_it')
    fake_s3_client = botocore.session.get_session().create_client('s3')
    with stub.Stubber(fake_s3_client) as stubber:
        stubber.add_response(method='head_bucket', service_response={})
        stubber.add_response(method='head_object', service_response={})
        mock_client.return_value = fake_s3_client
        self.assertRaises(exceptions.Duplicate, self.store.add, FAKE_UUID, image_s3, 0, self.hash_algo)