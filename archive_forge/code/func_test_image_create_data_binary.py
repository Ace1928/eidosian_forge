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
def test_image_create_data_binary(self):
    """Pass binary file-like object"""
    self.proxy.find_image = mock.Mock()
    self.proxy._upload_image = mock.Mock()
    data = io.BytesIO(b'\x00\x00')
    self.proxy.create_image(name='fake', data=data, validate_checksum=False, container='bare', disk_format='raw')
    self.proxy._upload_image.assert_called_with('fake', container_format='bare', disk_format='raw', filename=None, data=data, meta={}, properties={self.proxy._IMAGE_MD5_KEY: '', self.proxy._IMAGE_SHA256_KEY: '', self.proxy._IMAGE_OBJECT_KEY: 'bare/fake'}, timeout=3600, validate_checksum=False, use_import=False, stores=None, all_stores=None, all_stores_must_succeed=None, wait=False)