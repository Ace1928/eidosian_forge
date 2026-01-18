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
def ex_destroy_backendservice(self, backendservice):
    """
        Destroy a Backend Service.

        :param  backendservice: BackendService object to destroy
        :type   backendservice: :class:`GCEBackendService`

        :return:  True if successful
        :rtype:   ``bool``
        """
    request = '/global/backendServices/%s' % backendservice.name
    self.connection.async_request(request, method='DELETE')
    return True