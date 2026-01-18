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
def test_get_token_and_login_is_deprecated(self):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warnings.simplefilter('ignore', PendingDeprecationWarning)
        warnings.filterwarnings('ignore', '.*next release of cryptography')
        NoNetworkLaunchpad.get_token_and_login('consumer')
        self.assertEqual(str(caught[0].message), 'The Launchpad.get_token_and_login() method is deprecated. You should use Launchpad.login_anonymous() for anonymous access and Launchpad.login_with() for all other purposes.')
        self.assertEqual(caught[0].category, DeprecationWarning)