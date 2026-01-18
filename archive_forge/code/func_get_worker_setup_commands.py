import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
def get_worker_setup_commands(self, instance_type_name: str, num_successful_updates: int=0) -> List[str]:
    if num_successful_updates > 0 and self._node_config_provider.restart_only:
        return []
    return self.get_node_type_specific_config(instance_type_name, 'worker_setup_commands')