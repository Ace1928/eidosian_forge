import copy
import os
from typing import Any, Dict
from ray._private.utils import get_ray_temp_dir
from ray.autoscaler._private.cli_logger import cli_logger
def prepare_coordinator(config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    if 'max_workers' not in config:
        cli_logger.abort('The field `max_workers` is required when using an automatically managed on-premise cluster.')
    node_type = config['available_node_types'][LOCAL_CLUSTER_NODE_TYPE]
    node_type['min_workers'] = config.pop('min_workers', 0)
    node_type['max_workers'] = config['max_workers']
    return config