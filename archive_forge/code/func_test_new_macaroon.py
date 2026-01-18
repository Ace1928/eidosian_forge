import json
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
import six
from macaroonbakery.tests import common
from pymacaroons import serializers
def test_new_macaroon(self):
    m = bakery.Macaroon(b'rootkey', b'some id', 'here', bakery.LATEST_VERSION)
    self.assertIsNotNone(m)
    self.assertEqual(m._macaroon.identifier, b'some id')
    self.assertEqual(m._macaroon.location, 'here')
    self.assertEqual(m.version, bakery.LATEST_VERSION)