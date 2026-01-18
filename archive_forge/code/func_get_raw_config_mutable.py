import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
def get_raw_config_mutable(self) -> Dict[str, Any]:
    return self._node_configs