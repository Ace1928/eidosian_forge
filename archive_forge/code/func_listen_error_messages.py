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
def listen_error_messages(worker, threads_stopped):
    """Listen to error messages in the background on the driver.

    This runs in a separate thread on the driver and pushes (error, time)
    tuples to be published.

    Args:
        worker: The worker class that this thread belongs to.
        threads_stopped (threading.Event): A threading event used to signal to
            the thread that it should exit.
    """
    worker.gcs_error_subscriber.subscribe()
    try:
        if _internal_kv_initialized():
            error_message = _internal_kv_get(ray_constants.DEBUG_AUTOSCALING_ERROR)
            if error_message is not None:
                logger.warning(error_message.decode())
        while True:
            if threads_stopped.is_set():
                return
            _, error_data = worker.gcs_error_subscriber.poll()
            if error_data is None:
                continue
            if error_data['job_id'] not in [worker.current_job_id.binary(), JobID.nil().binary()]:
                continue
            error_message = error_data['error_message']
            if error_data['type'] == ray_constants.TASK_PUSH_ERROR:
                pass
            else:
                print_to_stdstream({'lines': [error_message], 'pid': 'raylet', 'is_err': False})
    except (OSError, ConnectionError) as e:
        logger.error(f'listen_error_messages: {e}')