import errno
import hashlib
import testtools
from unittest import mock
import ddt
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import images
def test_location_ops_when_server_disabled_location_ops(self):
    image_id = '3a4560a1-e585-443e-9b39-553b46ec92d1'
    estr = 'The administrator has disabled API access to image locations'
    url = 'http://bar.com/'
    meta = {'bar': 'barmeta'}
    e = self.assertRaises(exc.HTTPBadRequest, self.controller.delete_locations, image_id, set([url]))
    self.assertIn(estr, str(e))
    e = self.assertRaises(exc.HTTPBadRequest, self.controller.update_location, image_id, url, meta)
    self.assertIn(estr, str(e))