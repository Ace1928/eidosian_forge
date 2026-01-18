import time
from unittest import mock
import uuid
from pycadf import attachment
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import event
from pycadf import geolocation
from pycadf import host
from pycadf import identifier
from pycadf import measurement
from pycadf import metric
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import tag
from pycadf.tests import base
from pycadf import timestamp
def test_geolocation(self):
    geo = geolocation.Geolocation(id=identifier.generate_uuid(), latitude='43.6481 N', longitude='79.4042 W', elevation='0', accuracy='1', city='toronto', state='ontario', regionICANN='ca')
    self.assertEqual(True, geo.is_valid())
    dict_geo = geo.as_dict()
    for key in geolocation.GEO_KEYNAMES:
        self.assertIn(key, dict_geo)