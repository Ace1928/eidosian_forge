import logging
from ray._private.ray_constants import DEBUG_AUTOSCALING_STATUS_LEGACY
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
def legacy_log_info_string(autoscaler, nodes):
    tmp = 'Cluster status: '
    tmp += info_string(autoscaler, nodes)
    tmp += '\n'
    tmp += autoscaler.load_metrics.info_string()
    tmp += '\n'
    tmp += autoscaler.resource_demand_scheduler.debug_string(nodes, autoscaler.pending_launches.breakdown(), autoscaler.load_metrics.get_resource_utilization())
    if _internal_kv_initialized():
        _internal_kv_put(DEBUG_AUTOSCALING_STATUS_LEGACY, tmp, overwrite=True)
    logger.debug(tmp)