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
def test_load_auth_info(self):
    auth_info = agent.load_auth_info(self.agent_filename)
    self.assertEqual(str(auth_info.key), PRIVATE_KEY)
    self.assertEqual(str(auth_info.key.public_key), PUBLIC_KEY)
    self.assertEqual(auth_info.agents, [agent.Agent(url='https://1.example.com/', username='user-1'), agent.Agent(url='https://2.example.com/discharger', username='user-2'), agent.Agent(url='http://0.3.2.1', username='test-user')])