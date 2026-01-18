import datetime
import os
import sys
import mock
import pytest  # type: ignore
from six.moves import reload_module
from google.auth import _oauth2client
def test__convert_gce_app_assertion_credentials():
    old_credentials = oauth2client.contrib.gce.AppAssertionCredentials(email='some_email')
    new_credentials = _oauth2client._convert_gce_app_assertion_credentials(old_credentials)
    assert new_credentials.service_account_email == old_credentials.service_account_email