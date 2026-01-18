from openstack.image.v1 import _proxy
from openstack.image.v1 import image
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_image_upload_attrs(self):
    self.verify_create(self.proxy.upload_image, image.Image)