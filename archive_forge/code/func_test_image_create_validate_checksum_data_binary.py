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
def test_image_create_validate_checksum_data_binary(self):
    """Pass real data as binary"""
    self.proxy.find_image = mock.Mock()
    self.proxy._upload_image = mock.Mock()
    self.proxy.create_image(name='fake', data=b'fake', validate_checksum=True, container='bare', disk_format='raw')
    self.proxy.find_image.assert_called_with('fake')
    self.proxy._upload_image.assert_called_with('fake', container_format='bare', disk_format='raw', filename=None, data=b'fake', meta={}, properties={self.proxy._IMAGE_MD5_KEY: '144c9defac04969c7bfad8efaa8ea194', self.proxy._IMAGE_SHA256_KEY: 'b5d54c39e66671c9731b9f471e585d8262cd4f54963f0c93082d8dcf334d4c78', self.proxy._IMAGE_OBJECT_KEY: 'bare/fake'}, timeout=3600, validate_checksum=True, use_import=False, stores=None, all_stores=None, all_stores_must_succeed=None, wait=False)