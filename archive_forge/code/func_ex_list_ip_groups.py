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
def ex_list_ip_groups(self, details=False):
    """
        Lists IDs and names for shared IP groups.
        If details lists all details for shared IP groups.

        :param       details: True if details is required
        :type        details: ``bool``

        :rtype: ``list`` of :class:`OpenStack_1_0_SharedIpGroup`
        """
    uri = '/shared_ip_groups/detail' if details else '/shared_ip_groups'
    resp = self.connection.request(uri, method='GET')
    groups = findall(resp.object, 'sharedIpGroup', self.XML_NAMESPACE)
    return [self._to_shared_ip_group(el) for el in groups]