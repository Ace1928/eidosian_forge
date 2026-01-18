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
def load_and_execute_setup_hook_func(worker_process_setup_hook_key: str) -> Optional[str]:
    worker = ray._private.worker.global_worker
    assert worker.connected
    func_manager = worker.function_actor_manager
    try:
        worker_setup_func_info = func_manager.fetch_registered_method(_encode_function_key(worker_process_setup_hook_key), timeout=get_import_export_timeout())
    except Exception:
        error_message = f'Failed to import setup hook within {get_import_export_timeout()} seconds.\n{traceback.format_exc()}'
        return error_message
    try:
        setup_func = pickle.loads(worker_setup_func_info.function)
    except Exception:
        error_message = f'Failed to deserialize the setup hook method.\n{traceback.format_exc()}'
        return error_message
    try:
        setup_func()
    except Exception:
        error_message = f'Failed to execute the setup hook method. Function name:{worker_setup_func_info.function_name}\n{traceback.format_exc()}'
        return error_message
    return None