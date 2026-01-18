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
def ex_instancegroupmanager_set_autohealingpolicies(self, manager, healthcheck, initialdelaysec):
    """
        Set the Autohealing Policies for this Instance Group.

        :param  healthcheck: Healthcheck to add
        :type   healthcheck: :class:`GCEHealthCheck`

        :param  initialdelaysec:  The time to allow an instance to boot and
                                  applications to fully start before the first
                                  health check
        :type   initialdelaysec:  ``int``

        :return:  True if successful
        :rtype:   ``bool``
        """
    request_data = {}
    request_data['autoHealingPolicies'] = [{'healthCheck': healthcheck.path, 'initialDelaySec': initialdelaysec}]
    request = '/zones/{}/instanceGroupManagers/{}/'.format(manager.zone.name, manager.name)
    self.connection.async_request(request, method='PATCH', data=request_data)
    return True