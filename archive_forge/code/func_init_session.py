import functools
import logging
import os
import platform
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Type
import ray
from ray.air._internal.session import _get_session
from ray.air._internal.util import RunnerThread, StartTraceback
from ray.air.constants import (
from ray.data import Dataset
from ray.train import Checkpoint
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.storage import StorageContext
from ray.train.constants import (
from ray.train.error import SessionMisuseError
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.debug import log_once
from ray.util.placement_group import _valid_resource_shape
from ray.util.scheduling_strategies import (
def init_session(*args, **kwargs) -> None:
    global _session
    if _session:
        raise ValueError('A Train session is already in use. Do not call `init_session()` manually.')
    from ray import actor, remote_function
    if 'TUNE_DISABLE_RESOURCE_CHECKS' not in os.environ:
        actor._actor_launch_hook = _tune_task_and_actor_launch_hook
        remote_function._task_launch_hook = _tune_task_and_actor_launch_hook
    _session = _TrainSession(*args, **kwargs)