from openstack.cloud import _utils
from openstack import exceptions
from openstack.image.v2._proxy import Proxy
from openstack import utils
def get_image_id(self, image_name, exclude=None):
    image = self.get_image_exclude(image_name, exclude)
    if image:
        return image.id
    return None