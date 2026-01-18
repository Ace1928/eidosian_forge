import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_get_zone(self, name):
    """
        Return a Zone object based on the zone name.

        :param  name: The name of the zone.
        :type   name: ``str``

        :return:  A GCEZone object for the zone or None if not found
        :rtype:   :class:`GCEZone` or ``None``
        """
    if name.startswith('https://'):
        short_name = self._get_components_from_path(name)['name']
        request = name
    else:
        short_name = name
        request = '/zones/%s' % name
    if short_name in self.zone_dict:
        return self.zone_dict[short_name]
    try:
        response = self.connection.request(request, method='GET').object
    except ResourceNotFoundError:
        return None
    return self._to_zone(response)