import pytest
from google.auth import _helpers
from google.auth import exceptions
from google.auth import iam
from google.oauth2 import service_account
def test_iam_signer(http_request, credentials):
    credentials = credentials.with_scopes(['https://www.googleapis.com/auth/iam'])
    signer = iam.Signer(http_request, credentials, credentials.service_account_email)
    signed_blob = signer.sign('message')
    assert isinstance(signed_blob, bytes)