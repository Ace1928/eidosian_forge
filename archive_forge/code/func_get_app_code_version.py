import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Union
import ray
import ray.util.serialization_addons
from ray.serve._private.common import DeploymentID
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve.schema import ServeApplicationSchema
def get_app_code_version(app_config: ServeApplicationSchema) -> str:
    """Returns the code version of an application.

    Args:
        app_config: The application config.

    Returns: a hash of the import path and (application level) runtime env representing
            the code version of the application.
    """
    encoded = json.dumps({'import_path': app_config.import_path, 'runtime_env': app_config.runtime_env, 'args': app_config.args}, sort_keys=True).encode('utf-8')
    return hashlib.md5(encoded).hexdigest()