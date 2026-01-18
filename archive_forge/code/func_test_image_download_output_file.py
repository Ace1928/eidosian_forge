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
def test_image_download_output_file(self):
    sot = image.Image(**EXAMPLE)
    response = mock.Mock()
    response.status_code = 200
    response.iter_content.return_value = [b'01', b'02']
    response.headers = {'Content-MD5': calculate_md5_checksum(response.iter_content.return_value)}
    self.sess.get = mock.Mock(return_value=response)
    output_file = tempfile.NamedTemporaryFile()
    sot.download(self.sess, output=output_file.name)
    output_file.seek(0)
    self.assertEqual(b'0102', output_file.read())