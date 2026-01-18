import copy
import datetime
from functools import partial
import logging
from pathlib import Path
from pickle import PicklingError
import pprint as pp
import traceback
from typing import (
import ray
from ray.exceptions import RpcError
from ray.train import CheckpointConfig, SyncConfig
from ray.train._internal.storage import StorageContext
from ray.tune.error import TuneError
from ray.tune.registry import register_trainable, is_function_trainable
from ray.tune.stopper import CombinedStopper, FunctionStopper, Stopper, TimeoutStopper
from ray.util.annotations import DeveloperAPI, Deprecated
@property
def stopper(self):
    return self._stopper