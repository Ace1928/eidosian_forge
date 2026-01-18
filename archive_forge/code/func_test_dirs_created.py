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
def test_dirs_created(self):
    launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
    NoNetworkLaunchpad.login_with('not important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir)
    self.assertTrue(os.path.isdir(launchpadlib_dir))
    service_path = os.path.join(launchpadlib_dir, 'api.example.com')
    self.assertTrue(os.path.isdir(service_path))
    self.assertTrue(os.path.isdir(os.path.join(service_path, 'cache')))
    credentials_path = os.path.join(service_path, 'credentials')
    self.assertFalse(os.path.isdir(credentials_path))