import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import reload_module
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.auth.compute_engine import _metadata
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min)
def test_get_service_account_token_with_scopes_list(utcnow):
    ttl = 500
    request = make_request(json.dumps({'access_token': 'token', 'expires_in': ttl}), headers={'content-type': 'application/json'})
    token, expiry = _metadata.get_service_account_token(request, scopes=['foo', 'bar'])
    request.assert_called_once_with(method='GET', url=_metadata._METADATA_ROOT + PATH + '/token' + '?scopes=foo%2Cbar', headers=_metadata._METADATA_HEADERS)
    assert token == 'token'
    assert expiry == utcnow() + datetime.timedelta(seconds=ttl)