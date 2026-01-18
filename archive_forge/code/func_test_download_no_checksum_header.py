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
def test_download_no_checksum_header(self):
    sot = image.Image(**EXAMPLE)
    resp1 = FakeResponse(b'abc', headers={'Content-Type': 'application/octet-stream'})
    resp2 = FakeResponse({'checksum': '900150983cd24fb0d6963f7d28e17f72'})
    self.sess.get.side_effect = [resp1, resp2]
    rv = sot.download(self.sess)
    self.sess.get.assert_has_calls([mock.call('images/IDENTIFIER/file', stream=False), mock.call('images/IDENTIFIER', microversion=None, params={}, skip_cache=False)])
    self.assertEqual(rv, resp1)