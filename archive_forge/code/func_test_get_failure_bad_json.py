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
def test_get_failure_bad_json():
    request = make_request('{', headers={'content-type': 'application/json'})
    with pytest.raises(exceptions.TransportError) as excinfo:
        _metadata.get(request, PATH)
    assert excinfo.match('invalid JSON')
    request.assert_called_once_with(method='GET', url=_metadata._METADATA_ROOT + PATH, headers=_metadata._METADATA_HEADERS)