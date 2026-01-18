import dataclasses
import importlib
import logging
import json
import os
import yaml
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union
from pkg_resources import packaging
import ray
import ssl
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray.dashboard.modules.job.common import uri_to_http_components
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.utils import split_address
from ray.autoscaler._private.cli_logger import cli_logger
def parse_runtime_env_args(runtime_env: Optional[str]=None, runtime_env_json: Optional[str]=None, working_dir: Optional[str]=None):
    """
    Generates a runtime_env dictionary using `runtime_env`, `runtime_env_json`,
    and `working_dir` CLI options. Only one of `runtime_env` or
    `runtime_env_json` may be defined. `working_dir` overwrites the
    `working_dir` from any other option.
    """
    final_runtime_env = {}
    if runtime_env is not None:
        if runtime_env_json is not None:
            raise ValueError('Only one of --runtime_env and --runtime-env-json can be provided.')
        with open(runtime_env, 'r') as f:
            final_runtime_env = yaml.safe_load(f)
    elif runtime_env_json is not None:
        final_runtime_env = json.loads(runtime_env_json)
    if working_dir is not None:
        if 'working_dir' in final_runtime_env:
            cli_logger.warning('Overriding runtime_env working_dir with --working-dir option')
        final_runtime_env['working_dir'] = working_dir
    return final_runtime_env