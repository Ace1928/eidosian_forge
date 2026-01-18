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
def test_generate_form_signature_account_key(self, mock_expiration):
    self.register_uris([dict(method='HEAD', uri=self.container_endpoint, headers={'Content-Length': '0', 'X-Container-Object-Count': '0', 'Accept-Ranges': 'bytes', 'X-Storage-Policy': 'Policy-0', 'Date': 'Fri, 16 Dec 2016 18:29:05 GMT', 'X-Timestamp': '1481912480.41664', 'X-Trans-Id': 'tx60ec128d9dbf44b9add68-0058543271dfw1', 'X-Container-Bytes-Used': '0', 'Content-Type': 'text/plain; charset=utf-8'}), dict(method='HEAD', uri=self.endpoint + '/', headers={'X-Account-Meta-Temp-Url-Key': 'amazingly-secure-key'})])
    self.assertEqual((13345, '3cb9bc83d5a4136421bb2c1f58b963740566646f'), self.cloud.object_store.generate_form_signature(self.container, object_prefix='prefix/location', redirect_url='https://example.com/location', max_file_size=1024 * 1024 * 1024, max_upload_count=10, timeout=1000, temp_url_key=None))
    self.assert_calls()