import re
import uuid
import google.auth
from google.auth import downscoped
from google.auth.transport import requests
from google.cloud import exceptions
from google.cloud import storage
from google.oauth2 import credentials
import pytest
def test_downscoping(temp_blobs):
    """Tests token consumer access to cloud storage using downscoped tokens.

    Args:
        temp_blobs (Tuple[google.cloud.storage.blob.Blob, ...]): The temporarily
            created test cloud storage blobs (one readonly accessible, the other
            not).
    """
    accessible_blob, inaccessible_blob = temp_blobs
    bucket_name = accessible_blob.bucket.name

    def refresh_handler(request, scopes=None):
        return get_token_from_broker(bucket_name, _OBJECT_PREFIX)
    creds = credentials.Credentials(None, scopes=['https://www.googleapis.com/auth/cloud-platform'], refresh_handler=refresh_handler)
    storage_client = storage.Client(credentials=creds)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(accessible_blob.name)
    assert blob.download_as_bytes().decode('utf-8') == _ACCESSIBLE_CONTENT
    with pytest.raises(exceptions.Forbidden) as excinfo:
        blob.upload_from_string('Write operations are not allowed')
    assert excinfo.match('does not have storage.objects.create access')
    with pytest.raises(exceptions.Forbidden) as excinfo:
        bucket.blob(inaccessible_blob.name).download_as_bytes()
    assert excinfo.match('does not have storage.objects.get access')