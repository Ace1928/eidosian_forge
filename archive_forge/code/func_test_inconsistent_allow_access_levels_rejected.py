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
def test_inconsistent_allow_access_levels_rejected(self):
    """Catch an attempt to specify inconsistent allow_access_levels."""
    engine = NoNetworkAuthorizationEngine(SERVICE_ROOT, consumer_name='consumer', allow_access_levels=['FOO'])
    self.assertRaises(ValueError, NoNetworkLaunchpad.login_with, None, consumer_name='consumer', allow_access_levels=['BAR'], authorization_engine=engine)