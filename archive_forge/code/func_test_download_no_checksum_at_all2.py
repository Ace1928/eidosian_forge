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
def test_download_no_checksum_at_all2(self):
    sot = image.Image(**EXAMPLE)
    resp1 = FakeResponse(b'abc', headers={'Content-Type': 'application/octet-stream'})
    resp2 = FakeResponse({'checksum': None})
    self.sess.get.side_effect = [resp1, resp2]
    with self.assertLogs(logger='openstack', level='WARNING') as log:
        rv = sot.download(self.sess)
        self.assertEqual(len(log.records), 1, 'Too many warnings were logged')
        self.assertEqual('Unable to verify the integrity of image %s', log.records[0].msg)
        self.assertEqual((sot.id,), log.records[0].args)
    self.sess.get.assert_has_calls([mock.call('images/IDENTIFIER/file', stream=False), mock.call('images/IDENTIFIER', microversion=None, params={}, skip_cache=False)])
    self.assertEqual(rv, resp1)