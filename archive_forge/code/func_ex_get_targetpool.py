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
def ex_get_targetpool(self, name, region=None):
    """
        Return a TargetPool object based on a name and optional region.

        :param  name: The name of the target pool
        :type   name: ``str``

        :keyword  region: The region to search for the target pool in (set to
                          'all' to search all regions).
        :type     region: ``str`` or :class:`GCERegion` or ``None``

        :return:  A TargetPool object for the pool
        :rtype:   :class:`GCETargetPool`
        """
    region = self._set_region(region) or self._find_zone_or_region(name, 'targetPools', region=True, res_name='TargetPool')
    request = '/regions/{}/targetPools/{}'.format(region.name, name)
    response = self.connection.request(request, method='GET').object
    return self._to_targetpool(response)