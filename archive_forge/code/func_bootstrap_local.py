import copy
import os
from typing import Any, Dict
from ray._private.utils import get_ray_temp_dir
from ray.autoscaler._private.cli_logger import cli_logger
def bootstrap_local(config: Dict[str, Any]) -> Dict[str, Any]:
    return config