import atexit
import faulthandler
import functools
import inspect
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
import urllib
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
from urllib.parse import urlparse
import colorama
import setproctitle
from typing import Literal, Protocol
import ray
import ray._private.node
import ray._private.parameter
import ray._private.profiling as profiling
import ray._private.ray_constants as ray_constants
import ray._private.serialization as serialization
import ray._private.services as services
import ray._private.state
import ray._private.storage as storage
import ray.actor
import ray.cloudpickle as pickle  # noqa
import ray.job_config
import ray.remote_function
from ray import ActorID, JobID, Language, ObjectRef
from ray._raylet import raise_sys_exit_with_custom_error_message
from ray._raylet import ObjectRefGenerator, TaskID
from ray.runtime_env.runtime_env import _merge_runtime_env
from ray._private import ray_option_utils
from ray._private.client_mode_hook import client_mode_hook
from ray._private.function_manager import FunctionActorManager
from ray._private.inspect_util import is_cython
from ray._private.ray_logging import (
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray._private.runtime_env.setup_hook import (
from ray._private.storage import _load_class
from ray._private.utils import get_ray_doc_version
from ray.exceptions import ObjectStoreFullError, RayError, RaySystemError, RayTaskError
from ray.experimental.internal_kv import (
from ray.experimental import tqdm_ray
from ray.experimental.tqdm_ray import RAY_TQDM_MAGIC
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.util.debug import log_once
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.tracing.tracing_helper import _import_from_string
from ray.widgets import Template
from ray.widgets.util import repr_with_fallback
def print_logs(self):
    """Prints log messages from workers on all nodes in the same job."""
    subscriber = self.gcs_log_subscriber
    subscriber.subscribe()
    exception_type = ray.exceptions.RpcError
    localhost = services.get_node_ip_address()
    try:
        last_polling_batch_size = 0
        job_id_hex = self.current_job_id.hex()
        while True:
            if self.threads_stopped.is_set():
                return
            data = subscriber.poll()
            if data is None:
                last_polling_batch_size = 0
                continue
            if self._filter_logs_by_job and data['job'] and (data['job'] != job_id_hex):
                last_polling_batch_size = 0
                continue
            data['localhost'] = localhost
            global_worker_stdstream_dispatcher.emit(data)
            lagging = 100 <= last_polling_batch_size < subscriber.last_batch_size
            if lagging:
                logger.warning("The driver may not be able to keep up with the stdout/stderr of the workers. To avoid forwarding logs to the driver, use 'ray.init(log_to_driver=False)'.")
            last_polling_batch_size = subscriber.last_batch_size
    except (OSError, exception_type) as e:
        logger.error(f'print_logs: {e}')
    finally:
        subscriber.close()