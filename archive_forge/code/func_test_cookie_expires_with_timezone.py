import json
from datetime import datetime
from unittest import TestCase
import macaroonbakery.bakery as bakery
import pymacaroons
from macaroonbakery._utils import cookie
from pymacaroons.serializers import json_serializer
def test_cookie_expires_with_timezone(self):
    from datetime import tzinfo
    timestamp = datetime.utcnow().replace(tzinfo=tzinfo())
    self.assertRaises(ValueError, cookie, 'http://example.com', 'test', 'value', expires=timestamp)