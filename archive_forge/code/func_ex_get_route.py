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
def ex_get_route(self, name):
    """
        Return a Route object based on a route name.

        :param  name: The name of the route
        :type   name: ``str``

        :return:  A Route object for the named route
        :rtype:   :class:`GCERoute`
        """
    request = '/global/routes/%s' % name
    response = self.connection.request(request, method='GET').object
    return self._to_route(response)