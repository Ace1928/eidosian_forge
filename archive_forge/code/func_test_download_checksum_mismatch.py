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
def test_download_checksum_mismatch(self):
    sot = image.Image(**EXAMPLE)
    resp = FakeResponse(b'abc', headers={'Content-MD5': 'the wrong checksum', 'Content-Type': 'application/octet-stream'})
    self.sess.get.return_value = resp
    self.assertRaises(exceptions.InvalidResponse, sot.download, self.sess)