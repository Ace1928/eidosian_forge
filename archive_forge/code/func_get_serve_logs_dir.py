import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
def get_serve_logs_dir() -> str:
    """Get the directory that stores Serve log files."""
    return os.path.join(ray._private.worker._global_node.get_logs_dir_path(), 'serve')