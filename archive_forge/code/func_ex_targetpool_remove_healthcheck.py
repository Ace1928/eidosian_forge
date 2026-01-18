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
def ex_targetpool_remove_healthcheck(self, targetpool, healthcheck):
    """
        Remove a health check from a target pool.

        :param  targetpool: The targetpool to remove health check from
        :type   targetpool: ``str`` or :class:`GCETargetPool`

        :param  healthcheck: The healthcheck to remove
        :type   healthcheck: ``str`` or :class:`GCEHealthCheck`

        :return: True if successful
        :rtype:  ``bool``
        """
    if not hasattr(targetpool, 'name'):
        targetpool = self.ex_get_targetpool(targetpool)
    if not hasattr(healthcheck, 'name'):
        healthcheck = self.ex_get_healthcheck(healthcheck)
    targetpool_data = {'healthChecks': [{'healthCheck': healthcheck.extra['selfLink']}]}
    request = '/regions/{}/targetPools/{}/removeHealthCheck'.format(targetpool.region.name, targetpool.name)
    self.connection.async_request(request, method='POST', data=targetpool_data)
    index = None
    for i, hc in enumerate(targetpool.healthchecks):
        if hc.name == healthcheck.name:
            index = i
    if index is not None:
        targetpool.healthchecks.pop(index)
    return True