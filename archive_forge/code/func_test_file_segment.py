from hashlib import sha1
import random
import string
import tempfile
import time
from unittest import mock
import requests_mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack.object_store.v1 import account
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit.cloud import test_object as base_test_object
from openstack.tests.unit import test_proxy_base
def test_file_segment(self):
    file_size = 4200
    content = ''.join((random.choice(string.ascii_uppercase + string.digits) for _ in range(file_size))).encode('latin-1')
    self.imagefile = tempfile.NamedTemporaryFile(delete=False)
    self.imagefile.write(content)
    self.imagefile.close()
    segments = self.proxy._get_file_segments(endpoint='test_container/test_image', filename=self.imagefile.name, file_size=file_size, segment_size=1000)
    self.assertEqual(len(segments), 5)
    segment_content = b''
    for index, (name, segment) in enumerate(segments.items()):
        self.assertEqual('test_container/test_image/{index:0>6}'.format(index=index), name)
        segment_content += segment.read()
    self.assertEqual(content, segment_content)