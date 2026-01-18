import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
@property
def no_restart(self) -> bool:
    return self._node_configs.get('no_restart', False)