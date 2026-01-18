import copy
from typing import Dict, List, Literal, Optional, Union
import ray
from ray.actor import ActorHandle
from ray.data import DataIterator, Dataset, ExecutionOptions, NodeIdStr
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data.preprocessor import Preprocessor
from ray.train.constants import TRAIN_DATASET_KEY  # noqa
from ray.util.annotations import DeveloperAPI, PublicAPI
def set_train_total_resources(self, num_train_cpus: float, num_train_gpus: float):
    """Set the total number of CPUs and GPUs used by training.

        If CPU or GPU resource limits are not set, they will be set to the
        total cluster resources minus the resources used by training.
        """
    self._num_train_cpus = num_train_cpus
    self._num_train_gpus = num_train_gpus