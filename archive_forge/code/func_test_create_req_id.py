import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_create_req_id(self):
    params = {'id': '4', 'name': 'image-4', 'container_format': 'ovf', 'disk_format': 'vhd', 'owner': 'asdf', 'size': 1024, 'min_ram': 512, 'min_disk': 10, 'copy_from': 'http://example.com', 'properties': {'a': 'b', 'c': 'd'}, 'return_req_id': []}
    image = self.mgr.create(**params)
    expect_headers = {'x-image-meta-id': '4', 'x-image-meta-name': 'image-4', 'x-image-meta-container_format': 'ovf', 'x-image-meta-disk_format': 'vhd', 'x-image-meta-owner': 'asdf', 'x-image-meta-size': '1024', 'x-image-meta-min_ram': '512', 'x-image-meta-min_disk': '10', 'x-glance-api-copy-from': 'http://example.com', 'x-image-meta-property-a': 'b', 'x-image-meta-property-c': 'd'}
    expect = [('POST', '/v1/images', expect_headers, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual('1', image.id)
    expect_req_id = ['req-1234']
    self.assertEqual(expect_req_id, params['return_req_id'])