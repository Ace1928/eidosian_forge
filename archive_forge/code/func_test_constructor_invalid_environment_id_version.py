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
def test_constructor_invalid_environment_id_version(self):
    credential_source = self.CREDENTIAL_SOURCE.copy()
    credential_source['environment_id'] = 'aws3'
    with pytest.raises(ValueError) as excinfo:
        self.make_credentials(credential_source=credential_source)
    assert excinfo.match("aws version '3' is not supported in the current build.")