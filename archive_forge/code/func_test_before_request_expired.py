import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
from google.auth import transport
@mock.patch('google.auth._helpers.utcnow')
def test_before_request_expired(self, utcnow):
    headers = {}
    request = self.make_mock_request(status=http_client.OK, data=self.SUCCESS_RESPONSE)
    credentials = self.make_credentials()
    credentials.token = 'token'
    utcnow.return_value = datetime.datetime.min
    credentials.expiry = datetime.datetime.min + _helpers.REFRESH_THRESHOLD + datetime.timedelta(seconds=1)
    assert credentials.valid
    assert not credentials.expired
    credentials.before_request(request, 'POST', 'https://example.com/api', headers)
    assert headers == {'authorization': 'Bearer token'}
    utcnow.return_value = datetime.datetime.min + datetime.timedelta(seconds=1)
    assert not credentials.valid
    assert credentials.expired
    credentials.before_request(request, 'POST', 'https://example.com/api', headers)
    assert headers == {'authorization': 'Bearer {}'.format(self.SUCCESS_RESPONSE['access_token'])}