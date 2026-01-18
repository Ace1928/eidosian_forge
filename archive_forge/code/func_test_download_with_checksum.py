import http.client as http
import io
from unittest import mock
import uuid
from cursive import exception as cursive_exception
import glance_store
from glance_store._drivers import filesystem
from oslo_config import cfg
import webob
import glance.api.policy
import glance.api.v2.image_data
from glance.common import exception
from glance.common import wsgi
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_download_with_checksum(self):
    request = wsgi.Request.blank('/')
    request.environ = {}
    response = webob.Response()
    response.request = request
    checksum = '0745064918b49693cca64d6b6a13d28a'
    image = FakeImage(size=3, checksum=checksum, data=[b'Z', b'Z', b'Z'])
    self.serializer.download(response, image)
    self.assertEqual(b'ZZZ', response.body)
    self.assertEqual('3', response.headers['Content-Length'])
    self.assertEqual(checksum, response.headers['Content-MD5'])
    self.assertEqual('application/octet-stream', response.headers['Content-Type'])