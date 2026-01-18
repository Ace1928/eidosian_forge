import json
from datetime import datetime
from unittest import TestCase
import macaroonbakery.bakery as bakery
import pymacaroons
from macaroonbakery._utils import cookie
from pymacaroons.serializers import json_serializer
def test_cookie_expires_naive(self):
    timestamp = datetime.utcnow()
    c = cookie('http://example.com', 'test', 'value', expires=timestamp)
    self.assertEqual(c.expires, int((timestamp - datetime(1970, 1, 1)).total_seconds()))