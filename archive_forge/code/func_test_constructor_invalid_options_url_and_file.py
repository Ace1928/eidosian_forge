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
def test_constructor_invalid_options_url_and_file(self):
    credential_source = {'url': self.CREDENTIAL_URL, 'file': SUBJECT_TOKEN_TEXT_FILE}
    with pytest.raises(ValueError) as excinfo:
        self.make_credentials(credential_source=credential_source)
    assert excinfo.match('Ambiguous credential_source')