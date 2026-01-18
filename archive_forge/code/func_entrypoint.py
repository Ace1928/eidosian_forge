import inspect
import logging
import os
from functools import partial
from numbers import Number
from typing import Any, Callable, Dict, Optional, Type
from ray.air._internal.util import StartTraceback, RunnerThread
import queue
from ray.air.constants import (
import ray.train
from ray.train._internal.checkpoint_manager import _TrainingResult
from ray.train._internal.session import (
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.result import (
from ray.tune.trainable import Trainable
from ray.tune.utils import (
from ray.util.annotations import DeveloperAPI
from ray import tune
from ray import train, tune
from ray import tune
from ray import train, tune
def entrypoint():
    try:
        return self._trainable_func(self.config)
    except Exception as e:
        raise StartTraceback from e