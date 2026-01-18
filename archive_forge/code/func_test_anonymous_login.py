from contextlib import contextmanager
import os
import shutil
import socket
import stat
import tempfile
import unittest
import warnings
from lazr.restfulclient.resource import ServiceRoot
from launchpadlib.credentials import (
from launchpadlib import uris
import launchpadlib.launchpad
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_anonymous_login(self):
    """Test the anonymous login helper function."""
    launchpad = NoNetworkLaunchpad.login_anonymously('anonymous access', launchpadlib_dir=self.temp_dir, service_root=SERVICE_ROOT)
    self.assertEqual(launchpad.credentials.access_token.key, '')
    self.assertEqual(launchpad.credentials.access_token.secret, '')
    credentials_path = os.path.join(self.temp_dir, 'api.example.com', 'credentials', 'anonymous access')
    self.assertFalse(os.path.exists(credentials_path))