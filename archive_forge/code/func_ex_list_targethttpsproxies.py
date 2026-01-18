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
def ex_list_targethttpsproxies(self):
    """
        Return the list of target HTTPs proxies.

        :return:  A list of target https proxy objects
        :rtype:   ``list`` of :class:`GCETargetHttpsProxy`
        """
    request = '/global/targetHttpsProxies'
    response = self.connection.request(request, method='GET').object
    return [self._to_targethttpsproxy(x) for x in response.get('items', [])]