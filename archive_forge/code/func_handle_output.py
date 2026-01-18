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
def handle_output(output):
    if not output:
        return
    elif isinstance(output, dict):
        ray.train.report(output)
    elif isinstance(output, Number):
        ray.train.report({DEFAULT_METRIC: output})
    else:
        raise ValueError('Invalid return or yield value. Either return/yield a single number or a dictionary object in your trainable function.')