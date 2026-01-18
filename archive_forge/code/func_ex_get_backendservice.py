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
def ex_get_backendservice(self, name):
    """
        Return a Backend Service object based on name

        :param  name: The name of the backend service
        :type   name: ``str``

        :return:  A BackendService object for the backend service
        :rtype:   :class:`GCEBackendService`
        """
    request = '/global/backendServices/%s' % name
    response = self.connection.request(request, method='GET').object
    return self._to_backendservice(response)