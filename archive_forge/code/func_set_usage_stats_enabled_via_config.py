import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def set_usage_stats_enabled_via_config(enabled) -> None:
    config = {}
    try:
        with open(_usage_stats_config_path()) as f:
            config = json.load(f)
        if not isinstance(config, dict):
            logger.debug(f'Invalid ray config file, should be a json dict but got {type(config)}')
            config = {}
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f'Failed to load ray config file {e}')
    config['usage_stats'] = enabled
    try:
        os.makedirs(os.path.dirname(_usage_stats_config_path()), exist_ok=True)
        with open(_usage_stats_config_path(), 'w') as f:
            json.dump(config, f)
    except Exception as e:
        raise Exception(f'Failed to {('enable' if enabled else 'disable')} usage stats by writing {{"usage_stats": {('true' if enabled else 'false')}}} to {_usage_stats_config_path()}') from e