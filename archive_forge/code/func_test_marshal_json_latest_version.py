import json
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
import six
from macaroonbakery.tests import common
from pymacaroons import serializers
def test_marshal_json_latest_version(self):
    locator = bakery.ThirdPartyStore()
    bs = common.new_bakery('bs-loc', locator)
    ns = checkers.Namespace({'testns': 'x', 'otherns': 'y'})
    m = bakery.Macaroon(root_key=b'root key', id=b'id', location='location', version=bakery.LATEST_VERSION, namespace=ns)
    m.add_caveat(checkers.Caveat(location='bs-loc', condition='something'), bs.oven.key, locator)
    data = m.serialize_json()
    m1 = bakery.Macaroon.deserialize_json(data)
    self.assertEqual(m1.macaroon.signature, m.macaroon.signature)
    self.assertEqual(m1.macaroon.version, m.macaroon.version)
    self.assertEqual(len(m1.macaroon.caveats), 1)
    self.assertEqual(m1.namespace, m.namespace)
    self.assertEqual(m1._caveat_data, m._caveat_data)
    data = json.dumps(m, cls=bakery.MacaroonJSONEncoder)
    m1 = json.loads(data, cls=bakery.MacaroonJSONDecoder)
    self.assertEqual(m1.macaroon.signature, m.macaroon.signature)
    self.assertEqual(m1.macaroon.version, m.macaroon.version)
    self.assertEqual(len(m1.macaroon.caveats), 1)
    self.assertEqual(m1.namespace, m.namespace)
    self.assertEqual(m1._caveat_data, m._caveat_data)