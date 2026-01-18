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
def ex_destroy_autoscaler(self, autoscaler):
    """
        Destroy an Autoscaler.

        :param  autoscaler: Autoscaler object to destroy.
        :type   autoscaler: :class:`GCEAutoscaler`

        :return:  True if successful
        :rtype:   ``bool``
        """
    request = '/zones/{}/autoscalers/{}'.format(autoscaler.zone.name, autoscaler.name)
    self.connection.async_request(request, method='DELETE')
    return True