import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import ray
from ray._private.ray_constants import DEFAULT_RUNTIME_ENV_TIMEOUT_SECONDS
from ray._private.runtime_env.conda import get_uri as get_conda_uri
from ray._private.runtime_env.pip import get_uri as get_pip_uri
from ray._private.runtime_env.plugin_schema_manager import RuntimeEnvPluginSchemaManager
from ray._private.runtime_env.validation import OPTION_TO_VALIDATION_FN
from ray._private.thirdparty.dacite import from_dict
from ray.core.generated.runtime_env_common_pb2 import (
from ray.util.annotations import PublicAPI
@staticmethod
def parse_and_validate_runtime_env_config(config: Union[Dict, 'RuntimeEnvConfig']) -> 'RuntimeEnvConfig':
    if isinstance(config, RuntimeEnvConfig):
        return config
    elif isinstance(config, Dict):
        unknown_fields = set(config.keys()) - RuntimeEnvConfig.known_fields
        if len(unknown_fields):
            logger.warning(f'The following unknown entries in the runtime_env_config dictionary will be ignored: {unknown_fields}.')
        config_dict = dict()
        for field in RuntimeEnvConfig.known_fields:
            if field in config:
                config_dict[field] = config[field]
        return RuntimeEnvConfig(**config_dict)
    else:
        raise TypeError(f"runtime_env['config'] must be of type dict or RuntimeEnvConfig, got: {type(config)}")