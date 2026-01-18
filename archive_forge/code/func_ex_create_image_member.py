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
def ex_create_image_member(self, image_id, member_id):
    """
        Give a project access to an image.

        The image should have visibility status 'shared'.

        Note that this is not an idempotent operation. If this action is
        attempted using a tenant that is already in the image members
        group the API will throw a Conflict (409).
        See the 'create-image-member' section on
        https://developer.openstack.org/api-ref/image/v2/index.html

        :param str image_id: The ID of the image to share with the specified
        tenant
        :param str member_id: The ID of the project / tenant (the image member)
        Note that this is the Keystone project ID and not the project name,
        so something like e2151b1fe02d4a8a2d1f5fc331522c0a
        :return None:

        :param      image_id: ID of the image to share
        :type       image_id: ``str``

        :param      project: ID of the project to give access to the image
        :type       image_id: ``str``

        :rtype: ``list`` of :class:`NodeImageMember`
        """
    data = {'member': member_id}
    response = self.image_connection.request('/v2/images/%s/members' % image_id, method='POST', data=data)
    return self._to_image_member(response.object)