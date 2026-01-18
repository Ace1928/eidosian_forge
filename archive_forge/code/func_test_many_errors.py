from collections import deque
from json import dumps
import tempfile
import unittest
from launchpadlib.errors import Unauthorized
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.launchpad import (
from launchpadlib.testing.helpers import NoNetworkAuthorizationEngine
def test_many_errors(self):
    """We'll keep getting new tokens as long as tokens are the problem."""
    SimulatedResponsesLaunchpad.responses = [Response(401, b'Invalid token.'), Response(200, SIMPLE_WADL), Response(401, b'Expired token.'), Response(401, b'Invalid token.'), Response(200, SIMPLE_JSON)]
    self.assertEqual(self.engine.access_tokens_obtained, 0)
    SimulatedResponsesLaunchpad.login_with('application name', authorization_engine=self.engine)
    self.assertEqual(self.engine.access_tokens_obtained, 4)