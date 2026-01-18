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
@patch.object(NoNetworkLaunchpad, '_is_sudo', staticmethod(lambda: False))
def test_default_credentials_save_failed_is_to_raise_exception(self):
    launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
    service_root = 'http://api.example.com/'
    with fake_keyring(BadSaveKeyring()):
        self.assertRaises(RuntimeError, NoNetworkLaunchpad.login_with, 'not important', service_root=service_root, launchpadlib_dir=launchpadlib_dir)