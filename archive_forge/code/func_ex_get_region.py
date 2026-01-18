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
def ex_get_region(self, name):
    """
        Return a Region object based on the region name.

        :param  name: The name of the region.
        :type   name: ``str``

        :return:  A GCERegion object for the region
        :rtype:   :class:`GCERegion`
        """
    if name.startswith('https://'):
        short_name = self._get_components_from_path(name)['name']
        request = name
    else:
        short_name = name
        request = '/regions/%s' % name
    if short_name in self.region_dict:
        return self.region_dict[short_name]
    response = self.connection.request(request, method='GET').object
    return self._to_region(response)