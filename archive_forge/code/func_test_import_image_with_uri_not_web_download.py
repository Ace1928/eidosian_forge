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
def test_import_image_with_uri_not_web_download(self):
    sot = image.Image(**EXAMPLE)
    sot.import_image(self.sess, 'glance-direct')
    self.sess.post.assert_called_with('images/IDENTIFIER/import', headers={}, json={'method': {'name': 'glance-direct'}})