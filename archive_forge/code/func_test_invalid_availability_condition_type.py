import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import credentials
from google.auth import downscoped
from google.auth import exceptions
from google.auth import transport
def test_invalid_availability_condition_type(self):
    with pytest.raises(TypeError) as excinfo:
        make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, {'foo': 'bar'})
    assert excinfo.match("The provided availability_condition is not a 'google.auth.downscoped.AvailabilityCondition' or None.")