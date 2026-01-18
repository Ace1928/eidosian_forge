import boto
import tempfile
from boto.exception import InvalidUriError
from boto import storage_uri
from boto.compat import urllib
from boto.s3.keyfile import KeyFile
from tests.integration.s3.mock_storage_service import MockBucket
from tests.integration.s3.mock_storage_service import MockBucketStorageUri
from tests.integration.s3.mock_storage_service import MockConnection
from tests.unit import unittest
def test_invalid_scheme(self):
    uri_str = 'mars://bucket/object'
    try:
        boto.storage_uri(uri_str, validate=False, suppress_consec_slashes=False)
    except InvalidUriError as e:
        self.assertIn('Unrecognized scheme', e.message)