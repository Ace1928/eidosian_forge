import copy
import os
from typing import Any, Dict
from ray._private.utils import get_ray_temp_dir
from ray.autoscaler._private.cli_logger import cli_logger
def prepare_manual(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and sets defaults for configs of manually managed on-prem
    clusters.

    - Checks for presence of required `worker_ips` and `head_ips` fields.
    - Defaults min and max workers to the number of `worker_ips`.
    - Caps min and max workers at the number of `worker_ips`.
    - Writes min and max worker info into the single worker node type.
    """
    config = copy.deepcopy(config)
    if 'worker_ips' not in config['provider'] or 'head_ip' not in config['provider']:
        cli_logger.abort('Please supply a `head_ip` and list of `worker_ips`. Alternatively, supply a `coordinator_address`.')
    num_ips = len(config['provider']['worker_ips'])
    node_type = config['available_node_types'][LOCAL_CLUSTER_NODE_TYPE]
    config.setdefault('max_workers', num_ips)
    min_workers = config.pop('min_workers', num_ips)
    max_workers = config['max_workers']
    if min_workers > num_ips:
        cli_logger.warning(f'The value of `min_workers` supplied ({min_workers}) is greater than the number of available worker ips ({num_ips}). Setting `min_workers={num_ips}`.')
        node_type['min_workers'] = num_ips
    else:
        node_type['min_workers'] = min_workers
    if max_workers > num_ips:
        cli_logger.warning(f'The value of `max_workers` supplied ({max_workers}) is greater than the number of available worker ips ({num_ips}). Setting `max_workers={num_ips}`.')
        node_type['max_workers'] = num_ips
        config['max_workers'] = num_ips
    else:
        node_type['max_workers'] = max_workers
    if max_workers < num_ips:
        cli_logger.warning(f'The value of `max_workers` supplied ({max_workers}) is less than the number of available worker ips ({num_ips}). At most {max_workers} Ray worker nodes will connect to the cluster.')
    return config