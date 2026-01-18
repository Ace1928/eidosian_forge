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
def test_cannot_specify_both_filename_and_store(self):
    ignore, filename = tempfile.mkstemp()
    store = KeyringCredentialStore()
    self.assertRaises(ValueError, NoNetworkLaunchpad.login_with, application_name='not important', credentials_file=filename, credential_store=store)
    os.remove(filename)