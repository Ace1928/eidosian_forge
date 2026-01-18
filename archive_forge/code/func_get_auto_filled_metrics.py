import copy
from datetime import datetime
import logging
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import ray
import ray.cloudpickle as ray_pickle
from ray.air._internal.util import skip_exceptions, exception_cause
from ray.air.constants import (
from ray.train._internal.checkpoint_manager import _TrainingResult
from ray.train._internal.storage import StorageContext, _exists_at_fs_path
from ray.train import Checkpoint
from ray.tune.result import (
from ray.tune.utils import UtilMonitor
from ray.tune.utils.log import disable_ipython
from ray.tune.utils.util import Tee
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.util.annotations import DeveloperAPI, PublicAPI
def get_auto_filled_metrics(self, now: Optional[datetime]=None, time_this_iter: Optional[float]=None, timestamp: Optional[int]=None, debug_metrics_only: bool=False) -> dict:
    """Return a dict with metrics auto-filled by the trainable.

        If ``debug_metrics_only`` is True, only metrics that don't
        require at least one iteration will be returned
        (``ray.tune.result.DEBUG_METRICS``).
        """
    if now is None:
        now = datetime.today()
    autofilled = {TRIAL_ID: self.trial_id, 'date': now.strftime('%Y-%m-%d_%H-%M-%S'), 'timestamp': timestamp if timestamp else int(time.mktime(now.timetuple())), TIME_THIS_ITER_S: time_this_iter, TIME_TOTAL_S: self._time_total, PID: os.getpid(), HOSTNAME: platform.node(), NODE_IP: self._local_ip, 'config': self.config, 'time_since_restore': self._time_since_restore, 'iterations_since_restore': self._iterations_since_restore}
    if self._timesteps_since_restore:
        autofilled['timesteps_since_restore'] = self._timesteps_since_restore
    if debug_metrics_only:
        autofilled = {k: v for k, v in autofilled.items() if k in DEBUG_METRICS}
    return autofilled