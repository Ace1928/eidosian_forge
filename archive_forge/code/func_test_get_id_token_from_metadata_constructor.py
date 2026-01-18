import base64
import datetime
import mock
import pytest  # type: ignore
import responses  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.auth.compute_engine import credentials
from google.auth.transport import requests
def test_get_id_token_from_metadata_constructor(self):
    with pytest.raises(ValueError):
        credentials.IDTokenCredentials(mock.Mock(), 'audience', use_metadata_identity_endpoint=True, token_uri='token_uri')
    with pytest.raises(ValueError):
        credentials.IDTokenCredentials(mock.Mock(), 'audience', use_metadata_identity_endpoint=True, signer=mock.Mock())
    with pytest.raises(ValueError):
        credentials.IDTokenCredentials(mock.Mock(), 'audience', use_metadata_identity_endpoint=True, additional_claims={'key', 'value'})
    with pytest.raises(ValueError):
        credentials.IDTokenCredentials(mock.Mock(), 'audience', use_metadata_identity_endpoint=True, service_account_email='foo@example.com')