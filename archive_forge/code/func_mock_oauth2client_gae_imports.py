import datetime
import os
import sys
import mock
import pytest  # type: ignore
from six.moves import reload_module
from google.auth import _oauth2client
@pytest.fixture
def mock_oauth2client_gae_imports(mock_non_existent_module):
    mock_non_existent_module('google.appengine.api.app_identity')
    mock_non_existent_module('google.appengine.ext.ndb')
    mock_non_existent_module('google.appengine.ext.webapp.util')
    mock_non_existent_module('webapp2')