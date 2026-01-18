import io
from unittest import mock
import requests
from openstack import exceptions
from openstack.image.v2 import _proxy
from openstack.image.v2 import cache as _cache
from openstack.image.v2 import image as _image
from openstack.image.v2 import member as _member
from openstack.image.v2 import metadef_namespace as _metadef_namespace
from openstack.image.v2 import metadef_object as _metadef_object
from openstack.image.v2 import metadef_resource_type as _metadef_resource_type
from openstack.image.v2 import metadef_schema as _metadef_schema
from openstack.image.v2 import schema as _schema
from openstack.image.v2 import service_info as _service_info
from openstack.image.v2 import task as _task
from openstack import proxy as proxy_base
from openstack.tests.unit.image.v2 import test_image as fake_image
from openstack.tests.unit import test_proxy_base
@mock.patch.object(proxy_base.Proxy, '_get_resource')
def test_image_queue(self, mock_get_resource):
    fake_cache = _cache.Cache()
    mock_get_resource.return_value = fake_cache
    self._verify('openstack.image.v2.cache.Cache.queue', self.proxy.queue_image, method_args=['image-id'], expected_args=[self.proxy, 'image-id'])
    mock_get_resource.assert_called_once_with(_cache.Cache, None)