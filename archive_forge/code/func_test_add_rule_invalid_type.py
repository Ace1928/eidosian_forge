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
def test_add_rule_invalid_type(self):
    availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
    access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
    rules = [access_boundary_rule]
    credential_access_boundary = make_credential_access_boundary(rules)
    with pytest.raises(TypeError) as excinfo:
        credential_access_boundary.add_rule('invalid')
    assert excinfo.match("The provided rule does not contain a valid 'google.auth.downscoped.AccessBoundaryRule'.")
    assert len(credential_access_boundary.rules) == 1
    assert credential_access_boundary.rules[0] == access_boundary_rule