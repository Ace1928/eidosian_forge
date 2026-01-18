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
def ex_targetpool_set_backup_targetpool(self, targetpool, backup_targetpool, failover_ratio=0.1):
    """
        Set a backup targetpool.

        :param  targetpool: The existing primary targetpool
        :type   targetpool: :class:`GCETargetPool`

        :param  backup_targetpool: The existing targetpool to use for
                                   failover traffic.
        :type   backup_targetpool: :class:`GCETargetPool`

        :param  failover_ratio: The percentage of healthy VMs must fall at or
                                below this value before traffic will be sent
                                to the backup targetpool (default 0.10)
        :type   failover_ratio: ``float``

        :return:  True if successful
        :rtype:   ``bool``
        """
    region = targetpool.region.name
    name = targetpool.name
    req_data = {'target': backup_targetpool.extra['selfLink']}
    params = {'failoverRatio': failover_ratio}
    request = '/regions/{}/targetPools/{}/setBackup'.format(region, name)
    self.connection.async_request(request, method='POST', data=req_data, params=params)
    return True