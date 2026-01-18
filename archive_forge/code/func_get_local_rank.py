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
@PublicAPI(stability='beta')
@_warn_session_misuse(default_value=0)
def get_local_rank() -> int:
    """Get the local rank of this worker (rank of the worker on its node).

    .. testcode::

        import torch

        import ray
        from ray import train
        from ray.train import ScalingConfig
        from ray.train.torch import TorchTrainer

        def train_loop_per_worker(config):
            if torch.cuda.is_available():
                torch.cuda.set_device(train.get_context().get_local_rank())
            ...

        train_dataset = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")
        trainer = TorchTrainer(
            train_loop_per_worker,
            scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
            datasets={"train": train_dataset}
        )
        trainer.fit()

    .. testoutput::
        :hide:

        ...
    """
    session = _get_session()
    if not hasattr(session, 'local_rank'):
        raise RuntimeError('`get_local_rank` can only be called for TrainSession! Make sure you only use that in `train_loop_per_worker` functionthat is passed into `DataParallelTrainer`.')
    return session.local_rank