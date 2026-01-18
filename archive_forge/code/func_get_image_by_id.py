from openstack.cloud import _utils
from openstack import exceptions
from openstack.image.v2._proxy import Proxy
from openstack import utils
def get_image_by_id(self, id):
    """Get a image by ID

        :param id: ID of the image.
        :returns: An image :class:`openstack.image.v2.image.Image` object.
        """
    return self.image.get_image(id)