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
def test_authorization_engine_is_propagated(self):
    engine = NoNetworkAuthorizationEngine(SERVICE_ROOT, 'application name')
    NoNetworkLaunchpad.login_with(authorization_engine=engine)
    self.assertEqual(engine.request_tokens_obtained, 1)
    self.assertEqual(engine.access_tokens_obtained, 1)