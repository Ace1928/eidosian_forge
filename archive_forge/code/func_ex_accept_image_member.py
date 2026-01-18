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
def ex_accept_image_member(self, image_id, member_id):
    """
        Accept a pending image as a member.

        This call is idempotent unlike ex_create_image_member,
        you can accept the same image many times.

        :param      image_id: ID of the image to accept
        :type       image_id: ``str``

        :param      project: ID of the project to accept the image as
        :type       image_id: ``str``

        :rtype: ``bool``
        """
    data = {'status': 'accepted'}
    response = self.image_connection.request('/v2/images/{}/members/{}'.format(image_id, member_id), method='PUT', data=data)
    return self._to_image_member(response.object)