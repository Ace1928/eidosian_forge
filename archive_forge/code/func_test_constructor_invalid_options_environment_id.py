import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import identity_pool
from google.auth import transport
def test_constructor_invalid_options_environment_id(self):
    credential_source = {'url': self.CREDENTIAL_URL, 'environment_id': 'aws1'}
    with pytest.raises(ValueError) as excinfo:
        self.make_credentials(credential_source=credential_source)
    assert excinfo.match("Invalid Identity Pool credential_source field 'environment_id'")