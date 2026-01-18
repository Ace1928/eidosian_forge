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
def ex_create_targetpool(self, name, region=None, healthchecks=None, nodes=None, session_affinity=None, backup_pool=None, failover_ratio=None):
    """
        Create a target pool.

        :param  name: Name of target pool
        :type   name: ``str``

        :keyword  region: Region to create the target pool in. Defaults to
                          self.region
        :type     region: ``str`` or :class:`GCERegion` or ``None``

        :keyword  healthchecks: Optional list of health checks to attach
        :type     healthchecks: ``list`` of ``str`` or :class:`GCEHealthCheck`

        :keyword  nodes:  Optional list of nodes to attach to the pool
        :type     nodes:  ``list`` of ``str`` or :class:`Node`

        :keyword  session_affinity:  Optional algorithm to use for session
                                     affinity.
        :type     session_affinity:  ``str``

        :keyword  backup_pool: Optional backup targetpool to take over traffic
                               if the failover_ratio is exceeded.
        :type     backup_pool: ``GCETargetPool`` or ``None``

        :keyword  failover_ratio: The percentage of healthy VMs must fall at
                                  or below this value before traffic will be
                                  sent to the backup_pool.
        :type     failover_ratio: :class:`GCETargetPool` or ``None``

        :return:  Target Pool object
        :rtype:   :class:`GCETargetPool`
        """
    targetpool_data = {}
    region = region or self.region
    if backup_pool and (not failover_ratio):
        failover_ratio = 0.1
        targetpool_data['failoverRatio'] = failover_ratio
        targetpool_data['backupPool'] = backup_pool.extra['selfLink']
    if failover_ratio and (not backup_pool):
        e = 'Must supply a backup targetPool when setting failover_ratio'
        raise ValueError(e)
    targetpool_data['name'] = name
    if not hasattr(region, 'name'):
        region = self.ex_get_region(region)
    targetpool_data['region'] = region.extra['selfLink']
    if healthchecks:
        if not hasattr(healthchecks[0], 'name'):
            hc_list = [self.ex_get_healthcheck(h).extra['selfLink'] for h in healthchecks]
        else:
            hc_list = [h.extra['selfLink'] for h in healthchecks]
        targetpool_data['healthChecks'] = hc_list
    if nodes:
        if not hasattr(nodes[0], 'name'):
            node_list = [self.ex_get_node(n, 'all').extra['selfLink'] for n in nodes]
        else:
            node_list = [n.extra['selfLink'] for n in nodes]
        targetpool_data['instances'] = node_list
    if session_affinity:
        targetpool_data['sessionAffinity'] = session_affinity
    request = '/regions/%s/targetPools' % region.name
    self.connection.async_request(request, method='POST', data=targetpool_data)
    return self.ex_get_targetpool(name, region)