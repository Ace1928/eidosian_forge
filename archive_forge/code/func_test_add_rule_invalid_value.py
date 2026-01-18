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
def test_add_rule_invalid_value(self):
    availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
    access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
    rules = [access_boundary_rule] * 10
    credential_access_boundary = make_credential_access_boundary(rules)
    with pytest.raises(ValueError) as excinfo:
        credential_access_boundary.add_rule(access_boundary_rule)
    assert excinfo.match('Credential access boundary rules can have a maximum of 10 rules.')
    assert len(credential_access_boundary.rules) == 10