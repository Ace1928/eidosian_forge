from collections import deque
from json import dumps
import tempfile
import unittest
from launchpadlib.errors import Unauthorized
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.launchpad import (
from launchpadlib.testing.helpers import NoNetworkAuthorizationEngine
def test_good_token(self):
    """If our token is good, we never get another one."""
    SimulatedResponsesLaunchpad.responses = [Response(200, SIMPLE_WADL), Response(200, SIMPLE_JSON)]
    self.assertEqual(self.engine.access_tokens_obtained, 0)
    SimulatedResponsesLaunchpad.login_with('application name', authorization_engine=self.engine)
    self.assertEqual(self.engine.access_tokens_obtained, 1)