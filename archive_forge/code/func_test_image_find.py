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
def test_image_find(self):
    sot = image.Image()
    self.sess._get_connection = mock.Mock(return_value=self.cloud)
    self.sess.get.side_effect = [FakeResponse(None, 404, headers={}, reason='dummy'), FakeResponse({'images': []}), FakeResponse({'images': [EXAMPLE]})]
    result = sot.find(self.sess, EXAMPLE['name'])
    self.sess.get.assert_has_calls([mock.call('images/' + EXAMPLE['name'], microversion=None, params={}, skip_cache=False), mock.call('/images', headers={'Accept': 'application/json'}, microversion=None, params={'name': EXAMPLE['name']}), mock.call('/images', headers={'Accept': 'application/json'}, microversion=None, params={'os_hidden': True})])
    self.assertIsInstance(result, image.Image)
    self.assertEqual(IDENTIFIER, result.id)