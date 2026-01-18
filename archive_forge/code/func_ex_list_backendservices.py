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
def ex_list_backendservices(self):
    """
        Return a list of backend services.

        :return: A list of backend service objects.
        :rtype: ``list`` of :class:`GCEBackendService`
        """
    list_backendservices = []
    response = self.connection.request('/global/backendServices', method='GET').object
    list_backendservices = [self._to_backendservice(d) for d in response.get('items', [])]
    return list_backendservices