import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def test_get_meta_from_headers(self):
    resp = webob.Response()
    resp.headers = {'x-image-meta-name': 'test', 'x-image-meta-virtual-size': 80}
    result = utils.get_image_meta_from_headers(resp)
    self.assertEqual({'name': 'test', 'properties': {}, 'virtual_size': 80}, result)