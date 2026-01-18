import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
def prune_active_ips(self, active_ips: List[str]):
    """The Raylet ips stored by LoadMetrics are obtained by polling
        the GCS in Monitor.update_load_metrics().

        On the other hand, the autoscaler gets a list of node ips from
        its NodeProvider.

        This method removes from LoadMetrics the ips unknown to the autoscaler.

        Args:
            active_ips (List[str]): The node ips known to the autoscaler.
        """
    active_ips = set(active_ips)

    def prune(mapping, should_log):
        unwanted_ips = set(mapping) - active_ips
        for unwanted_ip in unwanted_ips:
            if should_log:
                logger.info(f'LoadMetrics: Removed ip: {unwanted_ip}.')
            del mapping[unwanted_ip]
        if unwanted_ips and should_log:
            logger.info('LoadMetrics: Removed {} stale ip mappings: {} not in {}'.format(len(unwanted_ips), unwanted_ips, active_ips))
        assert not unwanted_ips & set(mapping)
    prune(self.last_used_time_by_ip, should_log=True)
    prune(self.static_resources_by_ip, should_log=False)
    prune(self.raylet_id_by_ip, should_log=False)
    prune(self.dynamic_resources_by_ip, should_log=False)
    prune(self.last_heartbeat_time_by_ip, should_log=False)