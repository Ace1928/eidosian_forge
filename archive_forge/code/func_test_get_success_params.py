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
def test_get_success_params():
    data = 'foobar'
    request = make_request(data, headers={'content-type': 'text/plain'})
    params = {'recursive': 'true'}
    result = _metadata.get(request, PATH, params=params)
    request.assert_called_once_with(method='GET', url=_metadata._METADATA_ROOT + PATH + '?recursive=true', headers=_metadata._METADATA_HEADERS)
    assert result == data