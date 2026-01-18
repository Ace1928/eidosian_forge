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
@mock.patch('google.auth.compute_engine._metadata.get_service_account_info', autospec=True)
@mock.patch('google.auth.compute_engine._metadata.get', autospec=True)
def test_get_id_token_from_metadata(self, get, get_service_account_info):
    get.return_value = SAMPLE_ID_TOKEN
    get_service_account_info.return_value = {'email': 'foo@example.com'}
    cred = credentials.IDTokenCredentials(mock.Mock(), 'audience', use_metadata_identity_endpoint=True)
    cred.refresh(request=mock.Mock())
    assert cred.token == SAMPLE_ID_TOKEN
    assert cred.expiry == datetime.datetime.fromtimestamp(SAMPLE_ID_TOKEN_EXP)
    assert cred._use_metadata_identity_endpoint
    assert cred._signer is None
    assert cred._token_uri is None
    assert cred._service_account_email == 'foo@example.com'
    assert cred._target_audience == 'audience'
    with pytest.raises(ValueError):
        cred.sign_bytes(b'bytes')