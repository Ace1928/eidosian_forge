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
def ex_get_image_member(self, image_id, member_id):
    """
        Get a member of an image by id

        :param      image_id: ID of the image of which the member should
        be listed
        :type       image_id: ``str``

        :param      member_id: ID of the member to list
        :type       image_id: ``str``

        :rtype: ``list`` of :class:`NodeImageMember`
        """
    response = self.image_connection.request('/v2/images/{}/members/{}'.format(image_id, member_id))
    return self._to_image_member(response.object)