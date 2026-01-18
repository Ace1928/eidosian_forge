import tempfile
from unittest import mock
import testtools
import openstack.cloud.openstackcloud as oc_oc
from openstack import exceptions
from openstack.object_store.v1 import _proxy
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(_proxy, '_get_expiration', return_value=13345)
def test_generate_form_signature_key_argument(self, mock_expiration):
    self.assertEqual((13345, '1c283a05c6628274b732212d9a885265e6f67b63'), self.cloud.object_store.generate_form_signature(self.container, object_prefix='prefix/location', redirect_url='https://example.com/location', max_file_size=1024 * 1024 * 1024, max_upload_count=10, timeout=1000, temp_url_key='amazingly-secure-key'))
    self.assert_calls()