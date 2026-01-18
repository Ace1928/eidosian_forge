import datetime
import mock
import pytest  # type: ignore
from google.auth import app_engine
def test_key_id(self, app_identity):
    app_identity.sign_blob.return_value = (mock.sentinel.key_id, mock.sentinel.signature)
    signer = app_engine.Signer()
    assert signer.key_id is None