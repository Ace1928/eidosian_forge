import traceback
import logging
import base64
import os
from typing import Dict, Any, Callable, Union, Optional
import ray
import ray._private.ray_constants as ray_constants
from ray._private.storage import _load_class
import ray.cloudpickle as pickle
from ray.runtime_env import RuntimeEnv
def load_and_execute_setup_hook(worker_process_setup_hook_key: str) -> Optional[str]:
    """Load the setup hook from a given key and execute.

    Args:
        worker_process_setup_hook_key: The key to import the setup hook
            from GCS.
    Returns:
        An error message if it fails. None if it succeeds.
    """
    assert worker_process_setup_hook_key is not None
    if not worker_process_setup_hook_key.startswith(RUNTIME_ENV_FUNC_IDENTIFIER):
        return load_and_execute_setup_hook_module(worker_process_setup_hook_key)
    else:
        return load_and_execute_setup_hook_func(worker_process_setup_hook_key)