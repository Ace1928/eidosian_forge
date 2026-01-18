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
def test_components_of_application_key(self):
    launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
    keyring = InMemoryKeyring()
    service_root = 'http://api.example.com/'
    application_name = 'Super App 3000'
    with fake_keyring(keyring):
        launchpad = NoNetworkLaunchpad.login_with(application_name, service_root=service_root, launchpadlib_dir=launchpadlib_dir)
        consumer_name = launchpad.credentials.consumer.key
    application_key = list(keyring.data.keys())[0][1]
    self.assertIn(service_root, application_key)
    self.assertIn(consumer_name, application_key)
    self.assertEqual(application_key, consumer_name + '@' + service_root)