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
def set_autohealingpolicies(self, healthcheck, initialdelaysec):
    """
        Sets the autohealing policies for the instance for the instance group
        controlled by this manager.

        :param  healthcheck: Healthcheck to add
        :type   healthcheck: :class:`GCEHealthCheck`

        :param  initialdelaysec:  The time to allow an instance to boot and
                                  applications to fully start before the first
                                  health check
        :type   initialdelaysec:  ``int``

        :return:  Return True if successful.
        :rtype: ``bool``
        """
    return self.driver.ex_instancegroupmanager_set_autohealingpolicies(manager=self, healthcheck=healthcheck, initialdelaysec=initialdelaysec)