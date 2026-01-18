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
def ex_create_backend(self, instance_group, balancing_mode='UTILIZATION', max_utilization=None, max_rate=None, max_rate_per_instance=None, capacity_scaler=1, description=None):
    """
        Helper Object to create a backend.

        :param  instance_group: The Instance Group for this Backend.
        :type   instance_group: :class: `GCEInstanceGroup`

        :param  balancing_mode: Specifies the balancing mode for this backend.
                                For global HTTP(S) load balancing, the valid
                                values are UTILIZATION (default) and RATE.
                                For global SSL load balancing, the valid
                                values are UTILIZATION (default) and
                                CONNECTION.
        :type   balancing_mode: ``str``

        :param  max_utilization: Used when balancingMode is UTILIZATION.
                                 This ratio defines the CPU utilization
                                 target for the group. The default is 0.8.
                                 Valid range is [0.0, 1.0].
        :type   max_utilization: ``float``

        :param  max_rate: The max requests per second (RPS) of the group.
                          Can be used with either RATE or UTILIZATION balancing
                          modes, but required if RATE mode. For RATE mode,
                          either maxRate or maxRatePerInstance must be set.
        :type   max_rate: ``int``

        :param  max_rate_per_instance: The max requests per second (RPS) that
                                       a single backend instance can handle.
                                       This is used to calculate the capacity
                                       of the group. Can be used in either
                                       balancing mode. For RATE mode, either
                                       maxRate or maxRatePerInstance must be
                                       set.
        :type   max_rate_per_instance: ``float``

        :param  capacity_scaler: A multiplier applied to the group's maximum
                                 servicing capacity (based on UTILIZATION,
                                 RATE, or CONNECTION). Default value is 1,
                                 which means the group will serve up to 100%
                                 of its configured capacity (depending on
                                 balancingMode). A setting of 0 means the
                                 group is completely drained, offering 0%
                                 of its available capacity. Valid range is
                                 [0.0,1.0].
        :type   capacity_scaler: ``float``

        :param  description: An optional description of this resource.
                             Provide this property when you create the
                             resource.
        :type   description: ``str``

        :return: A GCEBackend object.
        :rtype: :class: `GCEBackend`
        """
    return GCEBackend(instance_group=instance_group, balancing_mode=balancing_mode, max_utilization=max_utilization, max_rate=max_rate, max_rate_per_instance=max_rate_per_instance, capacity_scaler=capacity_scaler, description=description)