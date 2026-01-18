import asyncio
import json
import logging
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple
from ray._private.ray_constants import (
import ray._private.runtime_env.agent.runtime_env_consts as runtime_env_consts
from ray._private.ray_logging import setup_component_logger
from ray._private.runtime_env.conda import CondaPlugin
from ray._private.runtime_env.container import ContainerManager
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.java_jars import JavaJarsPlugin
from ray._private.runtime_env.pip import PipPlugin
from ray._private.gcs_utils import GcsAioClient
from ray._private.runtime_env.plugin import (
from ray._private.utils import get_or_create_event_loop
from ray._private.runtime_env.plugin import RuntimeEnvPluginManager
from ray._private.runtime_env.py_modules import PyModulesPlugin
from ray._private.runtime_env.working_dir import WorkingDirPlugin
from ray._private.runtime_env.nsight import NsightPlugin
from ray._private.runtime_env.mpi import MPIPlugin
from ray.core.generated import (
from ray.core.generated.runtime_env_common_pb2 import (
from ray.runtime_env import RuntimeEnv, RuntimeEnvConfig
def get_or_create_logger(self, job_id: bytes):
    job_id = job_id.decode()
    if job_id not in self._per_job_logger_cache:
        params = self._logging_params.copy()
        params['filename'] = f'runtime_env_setup-{job_id}.log'
        params['logger_name'] = f'runtime_env_{job_id}'
        params['propagate'] = False
        per_job_logger = setup_component_logger(**params)
        self._per_job_logger_cache[job_id] = per_job_logger
    return self._per_job_logger_cache[job_id]