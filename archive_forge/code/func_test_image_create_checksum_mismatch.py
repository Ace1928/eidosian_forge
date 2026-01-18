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
def test_image_create_checksum_mismatch(self):
    fake_image = _image.Image(id='fake', properties={self.proxy._IMAGE_MD5_KEY: 'fake_md5', self.proxy._IMAGE_SHA256_KEY: 'fake_sha256'})
    self.proxy.find_image = mock.Mock(return_value=fake_image)
    self.proxy._upload_image = mock.Mock()
    self.proxy.create_image(name='fake', data=b'fake', md5='fake2_md5', sha256='fake2_sha256')
    self.proxy._upload_image.assert_called()