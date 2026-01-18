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
def filter_autoscaler_events(lines: List[str]) -> Iterator[str]:
    """Given raw log lines from the monitor, return only autoscaler events.

    For Autoscaler V1:
        Autoscaler events are denoted by the ":event_summary:" magic token.
    For Autoscaler V2:
        Autoscaler events are published from log_monitor.py which read
        them from the `event_AUTOSCALER.log`.
    """
    if not ray_constants.AUTOSCALER_EVENTS:
        return
    AUTOSCALER_LOG_FYI = 'Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.'

    def autoscaler_log_fyi_needed() -> bool:
        global autoscaler_log_fyi_printed
        if not autoscaler_log_fyi_printed:
            autoscaler_log_fyi_printed = True
            return True
        return False
    from ray.autoscaler.v2.utils import is_autoscaler_v2
    if is_autoscaler_v2():
        from ray._private.event.event_logger import parse_event, filter_event_by_level
        for event_line in lines:
            if autoscaler_log_fyi_needed():
                yield AUTOSCALER_LOG_FYI
            event = parse_event(event_line)
            if not event or not event.message:
                continue
            if filter_event_by_level(event, ray_constants.RAY_LOG_TO_DRIVER_EVENT_LEVEL):
                continue
            yield event.message
    else:
        for line in lines:
            if ray_constants.LOG_PREFIX_EVENT_SUMMARY in line:
                if autoscaler_log_fyi_needed():
                    yield AUTOSCALER_LOG_FYI
                yield line.split(ray_constants.LOG_PREFIX_EVENT_SUMMARY)[1]