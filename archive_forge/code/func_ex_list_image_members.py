import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def ex_list_image_members(self, image_id):
    """
        List all members of an image. See
        https://developer.openstack.org/api-ref/image/v2/index.html#sharing

        :param      image_id: ID of the image of which the members should
        be listed
        :type       image_id: ``str``

        :rtype: ``list`` of :class:`NodeImageMember`
        """
    response = self.image_connection.request('/v2/images/{}/members'.format(image_id))
    image_members = []
    for image_member in response.object['members']:
        image_members.append(self._to_image_member(image_member))
    return image_members