import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery.httpbakery.agent as agent
import requests.cookies
from httmock import HTTMock, response, urlmatch
from six.moves.urllib.parse import parse_qs, urlparse
def test_invalid_read_auth_info_arg(self):
    with self.assertRaises(agent.AgentFileFormatError):
        agent.read_auth_info(0)