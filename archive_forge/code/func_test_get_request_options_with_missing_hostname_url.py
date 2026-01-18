import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import aws
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
def test_get_request_options_with_missing_hostname_url(self):
    request_signer = aws.RequestSigner('us-east-2')
    with pytest.raises(ValueError) as excinfo:
        request_signer.get_request_options({'access_key_id': ACCESS_KEY_ID, 'secret_access_key': SECRET_ACCESS_KEY}, 'https://', 'POST')
    assert excinfo.match('Invalid AWS service URL')