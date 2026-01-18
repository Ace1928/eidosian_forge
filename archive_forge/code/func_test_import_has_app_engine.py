import datetime
import os
import sys
import mock
import pytest  # type: ignore
from six.moves import reload_module
from google.auth import _oauth2client
def test_import_has_app_engine(mock_oauth2client_gae_imports, reset__oauth2client_module):
    reload_module(_oauth2client)
    assert _oauth2client._HAS_APPENGINE