import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_customer_images(self, location=None):
    """
        Return a list of customer imported images

        :param location: The target location
        :type  location: :class:`NodeLocation` or ``str``

        :rtype: ``list`` of :class:`NodeImage`
        """
    params = {}
    if location is not None:
        params['datacenterId'] = self._location_to_location_id(location)
    return self._to_images(self.connection.request_with_orgId_api_2('image/customerImage', params=params).object, 'customerImage')