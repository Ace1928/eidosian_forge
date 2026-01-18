from collections import defaultdict, deque
from functools import partial
import pathlib
from typing import (
import uuid
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.learner.learner import LearnerSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.minibatch_utils import ShardBatchIterator
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.train._internal.backend_executor import BackendExecutor
from ray.tune.utils.file_transfer import sync_dir_between_nodes
from ray.util.annotations import PublicAPI
def set_is_module_trainable(self, is_module_trainable: Callable[[ModuleID, MultiAgentBatch], bool]=None) -> None:
    """Sets the function that determines whether a module is trainable.

        Args:
            is_module_trainable: A function that takes in a module id and a batch
                and returns a boolean indicating whether the module should be trained
                on the batch.
        """
    if is_module_trainable is not None:
        self._is_module_trainable = is_module_trainable