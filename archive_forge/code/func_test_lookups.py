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
def test_lookups(self):
    """Ensure that short service names turn into long service names."""
    with self.edge_deprecation_error():
        for alias in self.aliases:
            self.assertEqual(uris.lookup_service_root(alias), uris.service_roots[alias])
    with self.edge_deprecation_error():
        for alias in self.aliases:
            self.assertEqual(uris.lookup_web_root(alias), uris.web_roots[alias])
    other_root = 'http://some-other-server.com'
    self.assertEqual(uris.lookup_service_root(other_root), other_root)
    self.assertEqual(uris.lookup_web_root(other_root), other_root)
    not_a_url = 'not-a-url'
    self.assertRaises(ValueError, uris.lookup_service_root, not_a_url)
    self.assertRaises(ValueError, uris.lookup_web_root, not_a_url)