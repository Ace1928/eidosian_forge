import io
import operator
import tempfile
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import _log
from openstack import exceptions
from openstack.image.v2 import image
from openstack.tests.unit import base
from openstack import utils
def test_import_image_with_stores(self):
    sot = image.Image(**EXAMPLE)
    json = {'method': {'name': 'web-download', 'uri': 'such-a-good-uri'}, 'stores': ['ceph_1']}
    store = mock.MagicMock()
    store.id = 'ceph_1'
    sot.import_image(self.sess, 'web-download', uri='such-a-good-uri', stores=[store])
    self.sess.post.assert_called_with('images/IDENTIFIER/import', headers={}, json=json)