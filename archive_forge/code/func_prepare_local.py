import copy
import os
from typing import Any, Dict
from ray._private.utils import get_ray_temp_dir
from ray.autoscaler._private.cli_logger import cli_logger
def prepare_local(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare local cluster config for ingestion by cluster launcher and
    autoscaler.
    """
    config = copy.deepcopy(config)
    for field in ('head_node', 'worker_nodes', 'available_node_types'):
        if config.get(field):
            err_msg = unsupported_field_message.format(field)
            cli_logger.abort(err_msg)
    config['available_node_types'] = {LOCAL_CLUSTER_NODE_TYPE: {'node_config': {}, 'resources': {}}}
    config['head_node_type'] = LOCAL_CLUSTER_NODE_TYPE
    if 'coordinator_address' in config['provider']:
        config = prepare_coordinator(config)
    else:
        config = prepare_manual(config)
    return config