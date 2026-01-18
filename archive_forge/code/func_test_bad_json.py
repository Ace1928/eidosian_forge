from collections import deque
from json import dumps
import tempfile
import unittest
from launchpadlib.errors import Unauthorized
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.launchpad import (
from launchpadlib.testing.helpers import NoNetworkAuthorizationEngine
def test_bad_json(self):
    """Show that bad JSON causes an exception."""
    self.assertRaises(JSONDecodeError, self.launchpad_with_responses, Response(200, SIMPLE_WADL), Response(200, b'This is not JSON.'))