from openstack.cloud import _utils
from openstack import exceptions
from openstack.image.v2._proxy import Proxy
from openstack import utils
def get_image_exclude(self, name_or_id, exclude):
    for image in self.search_images(name_or_id):
        if exclude:
            if exclude not in image.name:
                return image
        else:
            return image
    return None