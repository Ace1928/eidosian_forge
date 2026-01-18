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
@pytest.mark.parametrize('region, time, credentials, original_request, signed_request', TEST_FIXTURES)
@mock.patch('google.auth._helpers.utcnow')
def test_get_request_options(self, utcnow, region, time, credentials, original_request, signed_request):
    utcnow.return_value = datetime.datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
    request_signer = aws.RequestSigner(region)
    actual_signed_request = request_signer.get_request_options(credentials, original_request.get('url'), original_request.get('method'), original_request.get('data'), original_request.get('headers'))
    assert actual_signed_request == signed_request